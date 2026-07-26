"""Production factories for the particle-view architecture and loss screens.

The runtime resolves every scientific row to authenticated target coordinates,
an exact frozen consumer checkpoint, split-scoped true-view loaders, and
same-consumer target-logit caches.  The caches are compact task-local
artifacts: they are never part of the deployable bundle and they cannot be
substituted across consumers, coordinates, or logical splits.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .campaign import _distillation_rows
from .consumer import ParticleViewConsumer, ParticleViewConsumerConfig
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .distillation import (
    DISTILLATION_LINEAGE_FIELDS,
    JOINT_LINEAGE_FIELDS,
    TARGET_LOGIT_CACHE_LINEAGE_FIELDS,
    DistillationTrainConfig,
    FrozenTargetLogitCache,
    JointFineTuneConfig,
    build_distillation_loss_screen,
    joint_schedule_contract_sha256,
    module_state_sha256,
    publish_target_logit_cache,
    validate_distillation_registration,
)
from .offline_teacher import LABEL_NAMES
from .post_target_runtime import (
    SelectedViewLoader,
    _artifact,
    _load_clean_consumer,
    _load_selected_cache_index,
    _materialize_selected_split,
    _pview_models,
    _selected_loader,
    _selected_target,
    _task_artifacts,
    _teacher_from_task,
    registered_target_for_run,
)
from .predictor import (
    NONSELECTABLE_PREDICTOR_ARCHITECTURES,
    PARTICLE_VIEW_PREDICTOR_ARCHITECTURES,
    HierarchicalParticleViewPredictor,
    build_predictor_architecture_config,
    count_unique_parameters,
)
from .registry import validate_particle_view_registry
from .runtime_data import (
    PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT,
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    validate_runtime_data_config,
)
from .target_runtime import CANONICAL_TARGET_DISCOVERY_RUN_ID
from .view_cache import load_particle_view_normalizer


PARTICLE_VIEW_DISTILLATION_FACTORY_CONFIG_CONTRACT = (
    "particle_view_distillation_factory_config_v1"
)
PARTICLE_VIEW_DISTILLATION_RUNTIME_BINDING_CONTRACT = (
    "particle_view_distillation_runtime_binding_v1"
)
TARGET_ALIASES = (
    "TARGET_CANONICAL_SELECTED",
    "TARGET_ALTERNATE_SELECTED",
)
DISTILLATION_CACHE_SPLITS = (
    "train",
    "model_val_stop",
    "model_val_select",
)


class _Limited:
    def __init__(self, loader, maximum: int | None) -> None:
        self.loader = loader
        self.maximum = maximum
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        if self.maximum is None:
            return iter(self.loader)
        from itertools import islice

        return islice(iter(self.loader), self.maximum)

    def __len__(self) -> int:
        if self.maximum is None:
            return len(self.loader)
        return min(len(self.loader), self.maximum)


class _FrozenSharedConsumerStem(nn.Module):
    """Reuse the exact consumer embedding while adapting to predictor width."""

    def __init__(
        self,
        embedding: nn.Module,
        *,
        consumer_width: int,
        predictor_width: int,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.projection = (
            nn.Identity()
            if consumer_width == predictor_width
            else nn.Linear(consumer_width, predictor_width)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Predictor input is [B,P,F]; Weaver embed consumes [B,F,P].
        embedded = self.embedding(values.transpose(1, 2))
        if isinstance(embedded, (tuple, list)):
            embedded = embedded[0]
        if embedded.ndim != 3:
            raise ValueError("shared consumer embedding must be rank three")
        if (
            embedded.shape[0] == values.shape[0]
            and embedded.shape[2] == values.shape[1]
        ):
            embedded = embedded.transpose(1, 2)
        elif embedded.shape[:2] != values.shape[:2]:
            raise ValueError("shared consumer embedding layout changed")
        return self.projection(embedded)


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # Validate eagerly so an invalid factory config fails before cache work.
    torch.device(value)
    return value


def build_distillation_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    validate_runtime_data_config(runtime_data_config, verify_cache_files=True)
    if not isinstance(device, str) or not device or int(num_workers) < 0:
        raise ValueError("distillation runtime settings are invalid")
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_val_batches", max_val_batches),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when set")
    campaign_rows = _distillation_rows()
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_FACTORY_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "runtime": {
                "device": device,
                "num_workers": int(num_workers),
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
            },
            "architecture_ids": list(
                PARTICLE_VIEW_PREDICTOR_ARCHITECTURES
            ),
            "distillation_rows_sha256": canonical_sha256(campaign_rows),
            "distillation_row_count": len(campaign_rows),
            "target_alias_policy": {
                "TARGET_CANONICAL_SELECTED": (
                    "predeclared_canonical_target"
                ),
                "TARGET_ALTERNATE_SELECTED": (
                    "rank_one_selected_target"
                ),
                "alias_collision": (
                    "record_warning_and_run_all_rows"
                ),
            },
            "predictor_initialization_policy": (
                "exact_pview0_when_structurally_compatible_else_seeded_fresh"
            ),
            "target_logit_cache_policy": (
                "task_local_float32_exact_same_consumer"
            ),
            "selected_splits": list(DISTILLATION_CACHE_SPLITS),
            "model_val_select_evaluations": 1,
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_distillation_factory_config(artifact)
    return artifact


def validate_distillation_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=PARTICLE_VIEW_DISTILLATION_FACTORY_CONFIG_CONTRACT,
    )
    expected = {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "runtime",
        "architecture_ids",
        "distillation_rows_sha256",
        "distillation_row_count",
        "target_alias_policy",
        "predictor_initialization_policy",
        "target_logit_cache_policy",
        "selected_splits",
        "model_val_select_evaluations",
        "stack_val_loaded",
        "final_test_loaded",
        "quality_warnings_stop_execution",
        "content_hash",
    }
    if set(payload) != expected:
        raise ValueError("distillation factory field inventory mismatch")
    data = payload["runtime_data_config"]
    validate_content_hash(
        data, expected_contract=PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT
    )
    runtime = payload["runtime"]
    if set(runtime) != {
        "device",
        "num_workers",
        "max_train_batches",
        "max_val_batches",
    }:
        raise ValueError("distillation runtime field inventory mismatch")
    rows = _distillation_rows()
    if (
        payload["runtime_data_config_sha256"] != data["content_hash"]
        or payload["architecture_ids"]
        != list(PARTICLE_VIEW_PREDICTOR_ARCHITECTURES)
        or payload["distillation_rows_sha256"]
        != canonical_sha256(rows)
        or payload["distillation_row_count"] != len(rows)
        or payload["selected_splits"] != list(DISTILLATION_CACHE_SPLITS)
        or payload["predictor_initialization_policy"]
        != "exact_pview0_when_structurally_compatible_else_seeded_fresh"
        or payload["target_logit_cache_policy"]
        != "task_local_float32_exact_same_consumer"
        or payload["model_val_select_evaluations"] != 1
        or payload["stack_val_loaded"] is not False
        or payload["final_test_loaded"] is not False
        or payload["quality_warnings_stop_execution"] is not False
        or not isinstance(runtime["device"], str)
        or not runtime["device"]
        or int(runtime["num_workers"]) < 0
        or any(
            value is not None and int(value) <= 0
            for value in (
                runtime["max_train_batches"],
                runtime["max_val_batches"],
            )
        )
    ):
        raise ValueError("distillation factory production policy changed")
    validate_runtime_data_config(data, verify_cache_files=False)
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "architecture_count": len(payload["architecture_ids"]),
        "distillation_row_count": payload["distillation_row_count"],
    }


def _config_from_payload(
    payload: Mapping[str, Any],
) -> ParticleViewConsumerConfig:
    names = {field.name for field in fields(ParticleViewConsumerConfig)}
    values = {name: payload[name] for name in names if name in payload}
    config = ParticleViewConsumerConfig(**values)
    if config.to_payload() != dict(payload):
        raise ValueError("consumer config payload changed")
    return config


def _load_screen_consumer(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    consumer_id: str,
) -> tuple[ParticleViewConsumer, dict[str, Any], Path]:
    artifacts = _task_artifacts(
        root, registry, f"SCREEN_{consumer_id}", seed
    )
    registration = load_hashed_json(
        _artifact(artifacts, "consumer_registration.json")
    )
    validate_content_hash(
        registration,
        expected_contract="particle_view_consumer_registration_v1",
    )
    checkpoint = _artifact(artifacts, "best_model_val_stop.pt")
    if registration["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("screen consumer checkpoint lineage mismatch")
    _, _, a0_model = _teacher_from_task(root, registry, "A0_VIEW", seed)
    model = ParticleViewConsumer(
        a0_model, _config_from_payload(registration["consumer_config"])
    )
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if (
        checkpoint_payload.get("role") != "Cview_probe"
        or checkpoint_payload.get("consumer_config")
        != model.config.to_payload()
    ):
        raise ValueError("screen consumer checkpoint contract mismatch")
    model.load_state_dict(
        checkpoint_payload["model_state_dict"], strict=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, registration, checkpoint


def _load_primary_robust_consumer(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    view_dim: int,
) -> tuple[ParticleViewConsumer, dict[str, Any], Path]:
    model, _, _ = _load_clean_consumer(
        root=root, registry=registry, seed=seed, view_dim=view_dim
    )
    artifacts = _task_artifacts(root, registry, "ROBUST_CONSUMER", seed)
    registration = load_hashed_json(
        _artifact(artifacts, "robust_consumer_registration.json")
    )
    validate_content_hash(
        registration,
        expected_contract="particle_view_robust_consumer_registration_v1",
    )
    checkpoint = _artifact(artifacts, "best_model_val_stop.pt")
    if (
        registration["checkpoint_sha256"] != sha256_file(checkpoint)
        or registration["consumer_config"] != model.config.to_payload()
    ):
        raise ValueError("robust consumer checkpoint/config mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, registration, checkpoint


def _load_primary_clean_consumer(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    view_dim: int,
) -> tuple[ParticleViewConsumer, dict[str, Any], Path]:
    model, registration, checkpoint = _load_clean_consumer(
        root=root, registry=registry, seed=seed, view_dim=view_dim
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, registration, checkpoint


def _target_for_alias(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    alias: str,
):
    selected = _selected_target(root=root, registry=registry, seed=seed)
    if alias == "TARGET_ALTERNATE_SELECTED":
        return selected, selected
    target_run_id = (
        CANONICAL_TARGET_DISCOVERY_RUN_ID
        if alias == "TARGET_CANONICAL_SELECTED"
        else alias
    )
    if not target_run_id.startswith("VGEN_"):
        raise ValueError("unknown target alias or registered target run")
    target = registered_target_for_run(
        root=root,
        registry=registry,
        run_id=target_run_id,
        seed=seed,
    )
    # Return both the resolved row and rank-one selection, since the latter
    # owns the immutable campaign-selection hash.
    return target, selected


def _target_loaders(
    *,
    data: Mapping[str, Any],
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    alias: str,
    runtime: Mapping[str, Any],
    target_override: Any | None = None,
):
    selected = _selected_target(root=root, registry=registry, seed=seed)
    target = (
        target_override
        if target_override is not None
        else _target_for_alias(
            root=root, registry=registry, seed=seed, alias=alias
        )[0]
    )
    index, publication, coordinate = _load_selected_cache_index(
        root, registry, seed
    )
    loaders = {}
    cache_loaders = {}
    aligned_by_split = {}
    view_resource_sha = {}
    manifests = {}
    selected_primary = target.run_id == selected.run_id
    for offset, split in enumerate(DISTILLATION_CACHE_SPLITS):
        maximum = (
            runtime["max_train_batches"]
            if split == "train"
            else runtime["max_val_batches"]
        )
        if selected_primary:
            loader, aligned, manifest = _selected_loader(
                data=data,
                index=index,
                split=split,
                batch_size=128,
                shuffle=split == "train",
                num_workers=runtime["num_workers"],
                seed=seed + offset,
                maximum_batches=maximum,
            )
            loaders[split] = loader
            cache_loaders[split] = getattr(
                loader, "full_loader", loader
            )
            aligned_by_split[split] = aligned
            view_resource_sha[split] = manifest["content_hash"]
            manifests[split] = manifest
            continue
        aligned = load_aligned_logical_jet_view(data, split)
        raw = _materialize_selected_split(
            selected=target,
            aligned=aligned,
            root=root,
            registry=registry,
            seed=seed,
            device=runtime["device"],
            num_workers=runtime["num_workers"],
        )
        normalizer = load_particle_view_normalizer(
            _artifact(target.artifacts, "provisional_normalizer.json")
        )
        normalized = raw.normalized(normalizer)
        base = make_logical_data_loader(
            aligned,
            mode="aligned",
            batch_size=128,
            shuffle=split == "train",
            num_workers=runtime["num_workers"],
            seed=seed + offset,
        )
        loaders[split] = _Limited(
            SelectedViewLoader(
                base,
                aligned=aligned,
                views=normalized.views.cpu().numpy(),
                mask=normalized.mask.cpu().numpy(),
            ),
            maximum,
        )
        cache_loaders[split] = loaders[split].loader
        aligned_by_split[split] = aligned
        view_resource_sha[split] = canonical_sha256(
            {
                "contract": "particle_view_live_target_resource_v1",
                "target_registration_sha256": target.registration[
                    "content_hash"
                ],
                "normalizer_sha256": normalizer.content_hash,
                "logical_split_sha256": aligned.logical_split_sha256,
                "ordered_identity_sha256": (
                    aligned.ordered_identity_sha256
                ),
            }
        )
        manifests[split] = None
    coordinate_sha = (
        coordinate["content_hash"]
        if selected_primary
        else canonical_sha256(
            {
                "contract": "particle_view_live_coordinate_reference_v1",
                "target_registration_sha256": target.registration[
                    "content_hash"
                ],
                "normalizer_sha256": load_particle_view_normalizer(
                    _artifact(
                        target.artifacts, "provisional_normalizer.json"
                    )
                ).content_hash,
            }
        )
    )
    return {
        "target": target,
        "selected": selected,
        "loaders": loaders,
        "cache_loaders": cache_loaders,
        "aligned": aligned_by_split,
        "view_resource_sha": view_resource_sha,
        "coordinate_sha256": coordinate_sha,
        "selected_primary": selected_primary,
        "selected_view_publication_sha256": publication["content_hash"],
        "cache_manifests": manifests,
    }


def _consumer_for_row(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    alias: str,
    consumer_id: str,
    view_dim: int,
    target_override: Any | None = None,
):
    if alias == "TARGET_ALTERNATE_SELECTED":
        if consumer_id == "C_ROBUST_MIX":
            return _load_primary_robust_consumer(
                root=root,
                registry=registry,
                seed=seed,
                view_dim=view_dim,
            )
        if consumer_id == "C_CLEAN":
            return _load_primary_clean_consumer(
                root=root,
                registry=registry,
                seed=seed,
                view_dim=view_dim,
            )
        raise ValueError(
            "alternate-target row requested an unregistered consumer"
        )
    if consumer_id == "C_TARGET_PROBE":
        target = (
            target_override
            if target_override is not None
            else _target_for_alias(
                root=root,
                registry=registry,
                seed=seed,
                alias=alias,
            )[0]
        )
        registration_path = _artifact(
            target.artifacts,
            "probe_consumer/consumer_registration.json",
        )
        checkpoint = _artifact(
            target.artifacts,
            "probe_consumer/best_model_val_stop.pt",
        )
        registration = load_hashed_json(registration_path)
        validate_content_hash(
            registration,
            expected_contract="particle_view_consumer_registration_v1",
        )
        if (
            registration["checkpoint_sha256"] != sha256_file(checkpoint)
            or int(registration["consumer_config"]["view_dim"]) != int(view_dim)
        ):
            raise ValueError("target probe consumer lineage/dimension mismatch")
        _, _, a0_model = _teacher_from_task(
            root, registry, "A0_VIEW", seed
        )
        model = ParticleViewConsumer(
            a0_model,
            _config_from_payload(registration["consumer_config"]),
        )
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, registration, checkpoint
    if consumer_id not in {
        "C_CLEAN",
        "C_ROBUST_MIX",
        "C_EMBED",
        "C_EMBED_PAIR",
    }:
        raise ValueError("canonical target requested an unknown consumer")
    return _load_screen_consumer(
        root=root,
        registry=registry,
        seed=seed,
        consumer_id=consumer_id,
    )


def _pview_registration_and_model(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    architecture_id: str,
    view_dim: int,
    consumer: ParticleViewConsumer,
) -> tuple[HierarchicalParticleViewPredictor, dict[str, Any], str]:
    artifacts = _task_artifacts(root, registry, "PVIEW0", seed)
    registration_path = _artifact(
        artifacts, "pview0_registration.json"
    )
    registration = load_hashed_json(registration_path)
    snapshots = _pview_models(registration, registration_path.parent)
    pview0 = snapshots[-1]
    requested = build_predictor_architecture_config(
        architecture_id, view_dim=view_dim
    )
    torch.manual_seed(int(seed))
    if requested.shared_consumer_stem:
        shared_stem = _FrozenSharedConsumerStem(
            consumer.a0_model.mod.embed,
            consumer_width=int(consumer.config.hidden_dim),
            predictor_width=int(requested.width),
        )
        model = HierarchicalParticleViewPredictor(
            requested, shared_particle_embedding=shared_stem
        )
        initialization = "seeded_fresh_with_frozen_shared_consumer_stem"
    else:
        model = HierarchicalParticleViewPredictor(requested)
        if (
            requested.structural_hash
            == pview0.config.structural_hash
        ):
            model.load_state_dict(pview0.state_dict(), strict=True)
            initialization = "exact_pview0_checkpoint"
        else:
            initialization = "seeded_fresh_architecture_specific"
    return model, registration, initialization


def _publish_task_target_logits(
    *,
    output: Path,
    consumer: nn.Module,
    consumer_registration: Mapping[str, Any],
    consumer_checkpoint: Path,
    target_resources: Mapping[str, Any],
    hlt_preprocessing_sha256: str,
    class_order_sha256: str,
    device: str,
) -> tuple[dict[str, FrozenTargetLogitCache], list[str]]:
    caches = {}
    artifacts = []
    for split in DISTILLATION_CACHE_SPLITS:
        aligned = target_resources["aligned"][split]
        lineage = {
            "split_identity_sha256": aligned.logical_split_sha256,
            "hlt_preprocessing_sha256": hlt_preprocessing_sha256,
            "class_order_sha256": class_order_sha256,
            "coordinate_binding_sha256": target_resources[
                "coordinate_sha256"
            ],
            "selected_view_cache_sha256": target_resources[
                "view_resource_sha"
            ][split],
            "consumer_registration_sha256": consumer_registration[
                "content_hash"
            ],
            "consumer_checkpoint_sha256": sha256_file(
                consumer_checkpoint
            ),
        }
        if set(lineage) != set(TARGET_LOGIT_CACHE_LINEAGE_FIELDS):
            raise RuntimeError("target-logit lineage inventory drifted")
        directory = output / "target_logits" / split
        manifest = (
            directory / f"{split}_frozen_consumer_logits.json"
        )
        data = directory / f"{split}_frozen_consumer_logits.npz"
        if manifest.exists() != data.exists():
            raise FileExistsError(
                "partial target-logit cache requires a fresh task attempt"
            )
        if not manifest.exists():
            publish_target_logit_cache(
                frozen_consumer=consumer,
                # Always cache the entire authorized logical split.  A
                # miniature/rehearsal max-batch limit may restrict optimizer
                # work, but must not make cache identity depend on the first
                # random shuffle encountered by the publisher.
                loader=target_resources["cache_loaders"][split],
                output_dir=directory,
                split=split,
                lineage=lineage,
                device=device,
            )
        cache = FrozenTargetLogitCache(manifest)
        if (
            cache.manifest["lineage"] != lineage
            or cache.manifest["split"] != split
        ):
            raise ValueError(
                "existing target-logit cache belongs to another task lineage"
            )
        cache.validate_consumer(consumer)
        caches[split] = cache
        artifacts.extend((str(data), str(manifest)))
    return caches, artifacts


def _distillation_row(run_id: str) -> dict[str, Any]:
    if not run_id.startswith("DISTILL_"):
        raise ValueError("distillation run ID is malformed")
    try:
        index = int(run_id.split("_", 1)[1])
        row = dict(_distillation_rows()[index])
    except (ValueError, IndexError) as exc:
        raise ValueError("distillation run ID is out of range") from exc
    return row


def _distillation_run_for(
    *,
    target_id: str,
    architecture_id: str,
    consumer_id: str,
    mode: str,
    loss_id: str,
) -> str:
    for index, row in enumerate(_distillation_rows()):
        if (
            row["target_id"] == target_id
            and row["architecture_id"] == architecture_id
            and row["consumer_id"] == consumer_id
            and row["mode"] == mode
            and row["loss_id"] == loss_id
        ):
            return f"DISTILL_{index:03d}"
    raise ValueError("required distillation source row is absent")


def _load_distillation_parent(
    *,
    root: Path,
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    artifacts = _task_artifacts(root, registry, run_id, seed)
    name = (
        "joint_registration.json"
        if _distillation_row(run_id)["mode"] != "frozen"
        else "distillation_registration.json"
    )
    registration_path = _artifact(artifacts, name)
    registration = load_hashed_json(registration_path)
    validated = validate_distillation_registration(
        registration, root=registration_path.parent
    )
    return registration, validated["checkpoint_payload"], registration_path


def _write_runtime_binding(
    *,
    output: Path,
    config: Mapping[str, Any],
    run_id: str,
    row: Mapping[str, Any],
    target_resources: Mapping[str, Any],
    consumer_registration: Mapping[str, Any],
    consumer_checkpoint: Path,
    caches: Mapping[str, FrozenTargetLogitCache],
    architecture_config: Mapping[str, Any],
    initialization: str,
) -> Path:
    alias_collision = (
        row["target_id"] == "TARGET_ALTERNATE_SELECTED"
        and target_resources["target"].run_id
        == CANONICAL_TARGET_DISCOVERY_RUN_ID
    )
    artifact = with_content_hash(
        {
            "contract": (
                PARTICLE_VIEW_DISTILLATION_RUNTIME_BINDING_CONTRACT
            ),
            "run_id": run_id,
            "campaign_row": dict(row),
            "factory_config_sha256": config["content_hash"],
            "resolved_target_run_id": target_resources["target"].run_id,
            "resolved_target_registration_sha256": target_resources[
                "target"
            ].registration["content_hash"],
            "rank_one_selected_target_run_id": target_resources[
                "selected"
            ].run_id,
            "target_alias_collision": alias_collision,
            "target_alias_warning": (
                "WARN_CANONICAL_AND_RANK_ONE_TARGET_ALIASES_COLLIDE"
                if alias_collision
                else None
            ),
            "consumer_registration_sha256": consumer_registration[
                "content_hash"
            ],
            "consumer_checkpoint_sha256": sha256_file(
                consumer_checkpoint
            ),
            "coordinate_binding_sha256": target_resources[
                "coordinate_sha256"
            ],
            "target_logit_cache_sha256_by_split": {
                split: caches[split].content_hash
                for split in DISTILLATION_CACHE_SPLITS
            },
            "architecture_config": dict(architecture_config),
            "architecture_config_sha256": canonical_sha256(
                architecture_config
            ),
            "predictor_initialization": initialization,
            "same_consumer_target_and_live": True,
            "quality_warning_stops_execution": False,
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    destination = output / "distillation_runtime_binding.json"
    write_immutable_json(destination, artifact)
    return destination


def build_distillation_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
    _row_override: Mapping[str, Any] | None = None,
    _target_override: Any | None = None,
    _force_fresh_predictor: bool = False,
) -> dict[str, Any]:
    validate_distillation_factory_config(config)
    validate_particle_view_registry(registry)
    rows = {row["run_id"]: row for row in registry["runs"]}
    if (
        run_id not in rows
        or int(seed) not in rows[run_id]["seed_ids"]
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("distillation task identity is invalid")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    runtime = dict(config["runtime"])
    runtime["device"] = _resolve_device(runtime["device"])
    data = config["runtime_data_config"]

    architecture_screen = run_id.startswith("ARCH_")
    if _row_override is not None:
        row = dict(_row_override)
        required = {
            "row_id",
            "target_id",
            "architecture_id",
            "consumer_id",
            "loss_id",
            "mode",
            "selectable",
            "privileged_claim_eligible",
        }
        if set(row) != required:
            raise ValueError("distillation row override inventory changed")
        expected_operation = (
            "frozen_distillation"
            if row["mode"] == "frozen"
            else "joint_finetuning"
        )
        if operation != expected_operation:
            raise ValueError("distillation row override operation changed")
    elif architecture_screen:
        architecture_id = run_id[5:]
        if architecture_id not in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES:
            raise ValueError("unknown architecture-screen run")
        if operation != "frozen_distillation":
            raise ValueError("architecture-screen operation changed")
        row = {
            "row_id": f"architecture={architecture_id}",
            "target_id": "TARGET_ALTERNATE_SELECTED",
            "architecture_id": architecture_id,
            "consumer_id": "C_ROBUST_MIX",
            "loss_id": "L_PRIMARY",
            "mode": "frozen",
            "selectable": (
                architecture_id
                not in NONSELECTABLE_PREDICTOR_ARCHITECTURES
            ),
            "privileged_claim_eligible": True,
        }
    else:
        row = _distillation_row(run_id)
        expected_operation = (
            "frozen_distillation"
            if row["mode"] == "frozen"
            else "joint_finetuning"
        )
        if operation != expected_operation:
            raise ValueError("distillation row operation changed")

    target_resources = _target_loaders(
        data=data,
        root=root,
        registry=registry,
        seed=seed,
        alias=row["target_id"],
        runtime=runtime,
        target_override=_target_override,
    )
    view_dim = int(
        target_resources["target"].registration["generator_config"][
            "bottleneck_width"
        ]
    )
    consumer, consumer_registration, consumer_checkpoint = (
        _consumer_for_row(
            root=root,
            registry=registry,
            seed=seed,
            alias=row["target_id"],
            consumer_id=row["consumer_id"],
            view_dim=view_dim,
            target_override=_target_override,
        )
    )
    predictor, pview_registration, initialization = (
        _pview_registration_and_model(
            root=root,
            registry=registry,
            seed=seed,
            architecture_id=row["architecture_id"],
            view_dim=view_dim,
            consumer=consumer,
        )
    )
    if _force_fresh_predictor:
        torch.manual_seed(int(seed))
        predictor = HierarchicalParticleViewPredictor(predictor.config)
        initialization = "seeded_fresh_control_initialization"
    a0_registration, _, _ = _teacher_from_task(
        root, registry, "A0_VIEW", seed
    )
    class_order_sha256 = canonical_sha256(list(LABEL_NAMES))
    caches, cache_artifacts = _publish_task_target_logits(
        output=output,
        consumer=consumer,
        consumer_registration=consumer_registration,
        consumer_checkpoint=consumer_checkpoint,
        target_resources=target_resources,
        hlt_preprocessing_sha256=a0_registration[
            "preprocessing_sha256"
        ],
        class_order_sha256=class_order_sha256,
        device=runtime["device"],
    )
    binding_path = _write_runtime_binding(
        output=output,
        config=config,
        run_id=run_id,
        row=row,
        target_resources=target_resources,
        consumer_registration=consumer_registration,
        consumer_checkpoint=consumer_checkpoint,
        caches=caches,
        architecture_config=predictor.config.to_payload(),
        initialization=initialization,
    )
    train = target_resources["aligned"]["train"]
    stop = target_resources["aligned"]["model_val_stop"]
    select = target_resources["aligned"]["model_val_select"]
    common_hashes = {
        "source_manifest_sha256": target_resources[
            "target"
        ].registration["source_manifest_sha256"],
        "train_identity_sha256": target_resources[
            "target"
        ].registration["train_identity_sha256"],
        "model_val_stop_split_sha256": stop.logical_split_sha256,
        "model_val_select_split_sha256": select.logical_split_sha256,
        "hlt_preprocessing_sha256": a0_registration[
            "preprocessing_sha256"
        ],
        "class_order_sha256": class_order_sha256,
        "coordinate_binding_sha256": target_resources[
            "coordinate_sha256"
        ],
    }
    base_artifacts = [*cache_artifacts, str(binding_path)]

    if row["mode"] == "frozen":
        objective = build_distillation_loss_screen()[row["loss_id"]]
        lineage = {
            **common_hashes,
            "target_selection_sha256": target_resources[
                "selected"
            ].selection["content_hash"],
            "pview0_registration_sha256": pview_registration[
                "content_hash"
            ],
            "initial_predictor_state_sha256": module_state_sha256(
                predictor
            ),
            "consumer_registration_sha256": consumer_registration[
                "content_hash"
            ],
            "consumer_checkpoint_sha256": sha256_file(
                consumer_checkpoint
            ),
            "train_target_logit_cache_sha256": caches[
                "train"
            ].content_hash,
            "model_val_stop_target_logit_cache_sha256": caches[
                "model_val_stop"
            ].content_hash,
            "model_val_select_target_logit_cache_sha256": caches[
                "model_val_select"
            ].content_hash,
        }
        if set(lineage) != set(DISTILLATION_LINEAGE_FIELDS):
            raise RuntimeError("distillation lineage inventory drifted")
        matched_updates = None
        schedule_source = None
        if objective.loss_id == "L_CE":
            source_run = _distillation_run_for(
                target_id=row["target_id"],
                architecture_id=row["architecture_id"],
                consumer_id=row["consumer_id"],
                mode="frozen",
                loss_id="L_PRIMARY",
            )
            source, _, _ = _load_distillation_parent(
                root=root,
                registry=registry,
                run_id=source_run,
                seed=seed,
            )
            matched_updates = int(source["optimizer_updates"])
            schedule_source = source["content_hash"]
        return {
            "kwargs": {
                "predictor": predictor,
                "frozen_consumer": consumer,
                "train_loader": target_resources["loaders"]["train"],
                "model_val_stop_loader": target_resources["loaders"][
                    "model_val_stop"
                ],
                "model_val_select_loader": target_resources["loaders"][
                    "model_val_select"
                ],
                "train_target_cache": caches["train"],
                "model_val_stop_target_cache": caches[
                    "model_val_stop"
                ],
                "model_val_select_target_cache": caches[
                    "model_val_select"
                ],
                "output_dir": str(output),
                "lineage": lineage,
                "objective": objective,
                "configuration_id": row["row_id"],
                "run_id": run_id,
                "deployed_parameters": count_unique_parameters(
                    (predictor, consumer)
                ),
                "config": DistillationTrainConfig(seed=seed),
                "matched_optimizer_updates": matched_updates,
                "schedule_match_source_sha256": schedule_source,
                "device": runtime["device"],
            },
            "artifact_paths": [
                *base_artifacts,
                str(output / "selected_distilled_predictor.pt"),
                str(output / "distillation_training_curves.json"),
                str(output / "distillation_generalization.json"),
                str(output / "distillation_registration.json"),
            ],
            "action": None,
        }

    primary_run = _distillation_run_for(
        target_id=row["target_id"],
        architecture_id=row["architecture_id"],
        consumer_id=row["consumer_id"],
        mode="frozen",
        loss_id="L_PRIMARY",
    )
    primary, primary_checkpoint, _ = _load_distillation_parent(
        root=root,
        registry=registry,
        run_id=primary_run,
        seed=seed,
    )
    predictor.load_state_dict(
        primary_checkpoint["predictor_state_dict"], strict=True
    )
    initialization = "selected_frozen_primary_bundle"
    privileged = row["mode"] == "joint"
    joint_config = JointFineTuneConfig(
        objective_id=(
            "JOINT_PRIVILEGED" if privileged else "JOINT_CE_ONLY"
        ),
        seed=seed,
    )
    lineage = {
        **common_hashes,
        "parent_distillation_registration_sha256": primary[
            "content_hash"
        ],
        "initial_predictor_state_sha256": module_state_sha256(
            predictor
        ),
        "initial_consumer_state_sha256": module_state_sha256(
            consumer
        ),
        "schedule_contract_sha256": joint_schedule_contract_sha256(
            joint_config
        ),
        "oracle_consumer_checkpoint_sha256": sha256_file(
            consumer_checkpoint
        ),
        "train_target_logit_cache_sha256": caches[
            "train"
        ].content_hash,
        "model_val_stop_target_logit_cache_sha256": caches[
            "model_val_stop"
        ].content_hash,
        "model_val_select_target_logit_cache_sha256": caches[
            "model_val_select"
        ].content_hash,
    }
    if set(lineage) != set(JOINT_LINEAGE_FIELDS):
        raise RuntimeError("joint lineage inventory drifted")
    matched_updates = None
    schedule_source = None
    if not privileged:
        source_run = _distillation_run_for(
            target_id=row["target_id"],
            architecture_id=row["architecture_id"],
            consumer_id=row["consumer_id"],
            mode="joint",
            loss_id="L_PRIMARY",
        )
        source, _, _ = _load_distillation_parent(
            root=root,
            registry=registry,
            run_id=source_run,
            seed=seed,
        )
        matched_updates = int(source["optimizer_updates"])
        schedule_source = source["content_hash"]
    # Rewrite the pre-action binding with no mutation is forbidden; its
    # initialization field describes the factory's architecture bootstrap,
    # while the joint parent is authenticated in JOINT_LINEAGE_FIELDS.
    return {
        "kwargs": {
            "predictor": predictor,
            "selected_frozen_consumer": consumer,
            "train_loader": target_resources["loaders"]["train"],
            "model_val_stop_loader": target_resources["loaders"][
                "model_val_stop"
            ],
            "model_val_select_loader": target_resources["loaders"][
                "model_val_select"
            ],
            "train_target_cache": caches["train"] if privileged else None,
            "model_val_stop_target_cache": (
                caches["model_val_stop"] if privileged else None
            ),
            "model_val_select_target_cache": (
                caches["model_val_select"] if privileged else None
            ),
            "output_dir": str(output),
            "lineage": lineage,
            "configuration_id": row["row_id"],
            "run_id": run_id,
            "config": joint_config,
            "matched_optimizer_updates": matched_updates,
            "schedule_match_source_sha256": schedule_source,
            "device": runtime["device"],
        },
        "artifact_paths": [
            *base_artifacts,
            str(output / "selected_joint_bundle.pt"),
            str(output / "joint_training_curves.json"),
            str(output / "joint_generalization.json"),
            str(output / "joint_registration.json"),
        ],
        "action": None,
    }


def build_distillation_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_distillation_factory_config(load_hashed_json(path))
    factory = (
        "teacher_logit_reco.local_particle_residual_field."
        "particle_view.distillation_runtime:build_distillation_factory"
    )
    common = {
        "factory": factory,
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    specs = {
        f"ARCH_{architecture_id}": {
            **common,
            "operation": "frozen_distillation",
        }
        for architecture_id in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES
    }
    for index, row in enumerate(_distillation_rows()):
        specs[f"DISTILL_{index:03d}"] = {
            **common,
            "operation": (
                "frozen_distillation"
                if row["mode"] == "frozen"
                else "joint_finetuning"
            ),
        }
    return specs


__all__ = [
    "DISTILLATION_CACHE_SPLITS",
    "PARTICLE_VIEW_DISTILLATION_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_RUNTIME_BINDING_CONTRACT",
    "TARGET_ALIASES",
    "build_distillation_factory",
    "build_distillation_factory_config",
    "build_distillation_task_specs",
    "validate_distillation_factory_config",
]
