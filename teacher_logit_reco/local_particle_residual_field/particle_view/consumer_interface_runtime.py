"""Production consumer-interface screen on the canonical particle-view target."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .campaign import CONSUMER_SCREEN_IDS
from .consumer import ParticleViewConsumer, ParticleViewConsumerConfig
from .consumer_train import (
    ParticleViewConsumerTrainConfig,
    evaluate_particle_view_consumer,
    train_particle_view_consumer,
)
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .post_target_runtime import (
    IndexedCandidateViewProvider,
    _artifact,
    _materialize_selected_split,
    _task_artifacts,
    _teacher_from_task,
    registered_target_for_run,
)
from .recovery_probe import FixedCapacityRecoveryProbe, RecoveryProbeConfig
from .registry import validate_particle_view_registry
from .runtime_data import (
    PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT,
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    validate_runtime_data_config,
)
from .target_runtime import (
    CANONICAL_TARGET_DISCOVERY_RUN_ID,
    CandidateViewSet,
    predict_recovery_probe_views,
)
from .view_cache import load_particle_view_normalizer


PARTICLE_VIEW_CONSUMER_SCREEN_FACTORY_CONFIG_CONTRACT = (
    "particle_view_consumer_screen_factory_config_v1"
)
PARTICLE_VIEW_CONSUMER_SCREEN_METRICS_CONTRACT = (
    "particle_view_consumer_screen_metrics_v1"
)
PARTICLE_VIEW_CONSUMER_INTERFACE_SELECTION_CONTRACT = (
    "particle_view_consumer_interface_selection_v1"
)


def _screen_recipe(
    consumer_id: str, *, blocks: int
) -> dict[str, Any]:
    if consumer_id not in CONSUMER_SCREEN_IDS:
        raise KeyError(f"unknown consumer screen {consumer_id!r}")
    midpoint = max(0, blocks // 2)
    values: dict[str, Any] = {
        "injection_block": 0,
        "view_path": "token_and_pair",
        "learned_trust": True,
        "augment_clean_view": False,
        "robust_probe_mixture": False,
    }
    if consumer_id == "C_RAWCAT":
        values.update(injection_block=-1, view_path="raw_projected")
    elif consumer_id == "C_EMBED":
        values.update(injection_block=-1, view_path="token_only")
    elif consumer_id == "C_PAIR":
        values.update(view_path="pair_only")
    elif consumer_id == "C_EMBED_PAIR":
        values.update(injection_block=-1)
    elif consumer_id == "C_INJECT0":
        values.update(injection_block=0)
    elif consumer_id == "C_INJECT1":
        values.update(injection_block=1)
    elif consumer_id == "C_INJECTMID":
        values.update(injection_block=midpoint)
    elif consumer_id == "C_FIXED_TRUST":
        values.update(learned_trust=False)
    elif consumer_id == "C_GATED_TRUST":
        pass
    elif consumer_id == "C_CLEAN":
        pass
    elif consumer_id == "C_DROPOUT":
        values.update(augment_clean_view=True)
    elif consumer_id == "C_ROBUST_MIX":
        values.update(robust_probe_mixture=True)
    return {
        "consumer_id": consumer_id,
        **values,
        "training_role": "Cview_probe",
        "epochs": 12,
        "selection_split": "model_val_select",
        "quality_gate_used": False,
    }


def build_consumer_screen_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    validate_runtime_data_config(runtime_data_config, verify_cache_files=True)
    if not isinstance(device, str) or not device or int(num_workers) < 0:
        raise ValueError("consumer-screen runtime is invalid")
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_val_batches", max_val_batches),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when set")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONSUMER_SCREEN_FACTORY_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "canonical_target_run_id": CANONICAL_TARGET_DISCOVERY_RUN_ID,
            "consumer_ids": list(CONSUMER_SCREEN_IDS),
            "runtime": {
                "device": device,
                "num_workers": int(num_workers),
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
            },
            "all_rows_complete_fixed_budget": True,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_consumer_screen_factory_config(artifact)
    return artifact


def validate_consumer_screen_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=PARTICLE_VIEW_CONSUMER_SCREEN_FACTORY_CONFIG_CONTRACT,
    )
    if set(payload) != {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "canonical_target_run_id",
        "consumer_ids",
        "runtime",
        "all_rows_complete_fixed_budget",
        "quality_warnings_stop_execution",
        "content_hash",
    }:
        raise ValueError("consumer-screen factory field inventory mismatch")
    data = payload["runtime_data_config"]
    validate_content_hash(
        data, expected_contract=PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT
    )
    if (
        payload["runtime_data_config_sha256"] != data["content_hash"]
        or payload["canonical_target_run_id"]
        != CANONICAL_TARGET_DISCOVERY_RUN_ID
        or payload["consumer_ids"] != list(CONSUMER_SCREEN_IDS)
        or payload["all_rows_complete_fixed_budget"] is not True
        or payload["quality_warnings_stop_execution"] is not False
    ):
        raise ValueError("consumer-screen production policy changed")
    validate_runtime_data_config(data, verify_cache_files=False)
    return {"ok": True, "content_hash": payload["content_hash"]}


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
        return (
            len(self.loader)
            if self.maximum is None
            else min(len(self.loader), self.maximum)
        )


class _RobustProbeViewProvider:
    """Deterministic candidate-screen proxy for the later robust mixture."""

    _schedule = (
        "true",
        "true",
        "true",
        "true",
        "true",
        "true",
        "predicted",
        "predicted",
        "predicted",
        "predicted",
        "predicted",
        "predicted",
        "predicted",
        "predicted",
        "predicted",
        "perturbed",
        "perturbed",
        "perturbed",
        "zero",
        "zero",
    )

    def __init__(
        self,
        true_views: CandidateViewSet,
        predicted_views: CandidateViewSet,
    ) -> None:
        if (
            true_views.logical_split_sha256
            != predicted_views.logical_split_sha256
            or not torch.equal(
                true_views.parent_indices, predicted_views.parent_indices
            )
        ):
            raise ValueError("robust probe view sets differ")
        self.true = true_views
        self.predicted = predicted_views
        self.calls = 0

    def __call__(self, batch: Mapping[str, Any]):
        positions = self.true.positions(batch["parent_indices"])
        device = batch["features"].device
        true = self.true.views[positions].to(device=device)
        predicted = self.predicted.views[positions].to(device=device)
        source = self._schedule[self.calls % len(self._schedule)]
        self.calls += 1
        if source == "true":
            view = true
        elif source == "predicted":
            view = predicted
        elif source == "zero":
            view = torch.zeros_like(true)
        else:
            residual_positions = (
                positions + self.calls
            ) % self.true.views.shape[0]
            residual = (
                self.true.views[residual_positions]
                - self.predicted.views[residual_positions]
            ).to(device=device)
            view = (predicted + residual).clamp(-6.0, 6.0)
        valid = batch["mask"][:, 0]
        view = torch.where(
            valid[:, :, None], view, torch.zeros_like(view)
        )
        return {"view": view, "raw_centered_view": view}


def _load_recovery_probe(
    selected,
    *,
    device: str,
) -> FixedCapacityRecoveryProbe:
    registration = load_hashed_json(
        _artifact(
            selected.artifacts, "recovery_probe_registration.json"
        )
    )
    raw = registration["config"]
    config = RecoveryProbeConfig(
        view_dim=int(raw["view_dim"]), seed=int(raw["seed"])
    )
    if config.to_payload() != raw:
        raise ValueError("canonical recovery-probe config changed")
    checkpoint_path = _artifact(
        selected.artifacts, "recovery_probe/best_model_val_stop.pt"
    )
    if sha256_file(checkpoint_path) != registration["checkpoint_sha256"]:
        raise ValueError("canonical recovery-probe checkpoint changed")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = FixedCapacityRecoveryProbe(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_consumer_interface_screen(
    *,
    model: ParticleViewConsumer,
    train_loader,
    stop_loader,
    select_loader,
    train_provider,
    stop_provider,
    select_provider,
    train_config: ParticleViewConsumerTrainConfig,
    lineage: Mapping[str, str],
    augment_clean_view: bool,
    consumer_id: str,
    recipe: Mapping[str, Any],
    output_dir: str,
    device: str,
) -> None:
    registration = train_particle_view_consumer(
        model=model,
        train_loader=train_loader,
        model_val_stop_loader=stop_loader,
        config=train_config,
        output_dir=output_dir,
        lineage=lineage,
        view_provider=train_provider,
        validation_view_provider=stop_provider,
        augment_clean_view_override=augment_clean_view,
        device=device,
    )
    checkpoint = torch.load(
        Path(output_dir) / "best_model_val_stop.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = evaluate_particle_view_consumer(
        model,
        select_loader,
        device=device,
        view_provider=select_provider,
    )
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONSUMER_SCREEN_METRICS_CONTRACT,
            "consumer_id": consumer_id,
            "run_id": f"SCREEN_{consumer_id}",
            "recipe": dict(recipe),
            "recipe_sha256": canonical_sha256(recipe),
            "consumer_registration_sha256": registration["content_hash"],
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "model_val_select": metrics,
            "ranking_rule": [
                "highest_accuracy",
                "lowest_cross_entropy",
                "lexicographic_consumer_id",
            ],
            "quality_gate_used": False,
            "stops_execution": False,
        }
    )
    write_immutable_json(
        Path(output_dir) / "consumer_interface_metrics.json", artifact
    )


def build_consumer_screen_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    if operation != "consumer_interface_screen":
        raise ValueError("consumer-screen operation changed")
    validate_consumer_screen_factory_config(config)
    validate_particle_view_registry(registry)
    if (
        not run_id.startswith("SCREEN_")
        or run_id[7:] not in CONSUMER_SCREEN_IDS
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("consumer-screen task identity is invalid")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    selected = registered_target_for_run(
        root=root,
        registry=registry,
        run_id=CANONICAL_TARGET_DISCOVERY_RUN_ID,
        seed=seed,
    )
    data = config["runtime_data_config"]
    runtime = config["runtime"]
    aligned = {
        split: load_aligned_logical_jet_view(data, split)
        for split in ("train", "model_val_stop", "model_val_select")
    }
    raw_views = {
        split: _materialize_selected_split(
            selected=selected,
            aligned=value,
            root=root,
            registry=registry,
            seed=seed,
            device=runtime["device"],
            num_workers=runtime["num_workers"],
        )
        for split, value in aligned.items()
    }
    normalizer = load_particle_view_normalizer(
        _artifact(selected.artifacts, "provisional_normalizer.json")
    )
    views = {
        split: value.normalized(normalizer)
        for split, value in raw_views.items()
    }
    del raw_views
    consumer_id = run_id[7:]
    a0_registration, _, a0_model = _teacher_from_task(
        root, registry, "A0_VIEW", seed
    )
    architecture = a0_registration["recipe"]["architecture"]
    recipe = _screen_recipe(
        consumer_id, blocks=int(architecture["num_layers"])
    )
    consumer_config = ParticleViewConsumerConfig(
        view_dim=int(
            selected.registration["generator_config"]["bottleneck_width"]
        ),
        hidden_dim=int(architecture["embed_dims"][-1]),
        num_heads=int(architecture["num_heads"]),
        injection_block=int(recipe["injection_block"]),
        view_path=str(recipe["view_path"]),
        learned_trust=bool(recipe["learned_trust"]),
    )
    for parameter in a0_model.parameters():
        parameter.requires_grad_(True)
    model = ParticleViewConsumer(a0_model, consumer_config)
    providers = {
        split: IndexedCandidateViewProvider(value)
        for split, value in views.items()
    }
    if recipe["robust_probe_mixture"]:
        probe = _load_recovery_probe(
            selected, device=runtime["device"]
        )
        predicted = {
            split: predict_recovery_probe_views(
                model=probe,
                aligned=aligned[split],
                true_views=views[split],
                device=runtime["device"],
                num_workers=runtime["num_workers"],
            )
            for split in aligned
        }
        providers = {
            split: _RobustProbeViewProvider(
                views[split], predicted[split]
            )
            for split in aligned
        }
    loaders = {}
    for split in aligned:
        maximum = (
            runtime["max_train_batches"]
            if split == "train"
            else runtime["max_val_batches"]
        )
        loaders[split] = _Limited(
            make_logical_data_loader(
                aligned[split],
                mode="aligned",
                batch_size=128,
                shuffle=split == "train",
                num_workers=runtime["num_workers"],
                seed=seed + (0 if split == "train" else 1),
            ),
            maximum,
        )
    return {
        "kwargs": {
            "model": model,
            "train_loader": loaders["train"],
            "stop_loader": loaders["model_val_stop"],
            "select_loader": loaders["model_val_select"],
            "train_provider": providers["train"],
            "stop_provider": providers["model_val_stop"],
            "select_provider": providers["model_val_select"],
            "train_config": ParticleViewConsumerTrainConfig.for_role(
                "Cview_probe", seed=seed
            ),
            "lineage": {
                "a0_registration_sha256": a0_registration["content_hash"],
                "target_registration_sha256": selected.registration[
                    "content_hash"
                ],
                "train_identity_sha256": selected.registration[
                    "train_identity_sha256"
                ],
                "model_val_stop_split_sha256": aligned[
                    "model_val_stop"
                ].logical_split_sha256,
                "normalizer_sha256": normalizer.content_hash,
            },
            "augment_clean_view": bool(recipe["augment_clean_view"]),
            "consumer_id": consumer_id,
            "recipe": recipe,
            "output_dir": str(output),
            "device": runtime["device"],
        },
        "artifact_paths": [
            str(output / "best_model_val_stop.pt"),
            str(output / "consumer_registration.json"),
            str(output / "training_curves.json"),
            str(output / "consumer_interface_metrics.json"),
        ],
        "action": None,
    }


def build_consumer_screen_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_consumer_screen_factory_config(load_hashed_json(path))
    return {
        f"SCREEN_{consumer_id}": {
            "operation": "consumer_interface_screen",
            "factory": (
                "teacher_logit_reco.local_particle_residual_field."
                "particle_view.consumer_interface_runtime:"
                "build_consumer_screen_factory"
            ),
            "factory_config_path": str(path),
            "factory_config_sha256": sha256_file(path),
        }
        for consumer_id in CONSUMER_SCREEN_IDS
    }


def select_consumer_interface(
    metrics: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(metrics) != len(CONSUMER_SCREEN_IDS):
        raise ValueError("consumer-interface metric coverage mismatch")
    by_id = {}
    for row in metrics:
        validate_content_hash(
            row, expected_contract=PARTICLE_VIEW_CONSUMER_SCREEN_METRICS_CONTRACT
        )
        by_id[row["consumer_id"]] = row
    if set(by_id) != set(CONSUMER_SCREEN_IDS):
        raise ValueError("consumer-interface ID coverage mismatch")
    ranked = sorted(
        by_id.values(),
        key=lambda row: (
            -float(row["model_val_select"]["accuracy"]),
            float(row["model_val_select"]["cross_entropy"]),
            str(row["consumer_id"]),
        ),
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONSUMER_INTERFACE_SELECTION_CONTRACT,
            "ranking_split": "model_val_select",
            "ranking_rule": [
                "highest_accuracy",
                "lowest_cross_entropy",
                "lexicographic_consumer_id",
            ],
            "ranked_consumer_ids": [
                row["consumer_id"] for row in ranked
            ],
            "selected_consumer_id": ranked[0]["consumer_id"],
            "selected_recipe": ranked[0]["recipe"],
            "selected_metric_sha256": ranked[0]["content_hash"],
            "quality_threshold_used_as_gate": False,
        }
    )


__all__ = [
    "PARTICLE_VIEW_CONSUMER_INTERFACE_SELECTION_CONTRACT",
    "PARTICLE_VIEW_CONSUMER_SCREEN_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_CONSUMER_SCREEN_METRICS_CONTRACT",
    "build_consumer_screen_factory",
    "build_consumer_screen_factory_config",
    "build_consumer_screen_task_specs",
    "run_consumer_interface_screen",
    "select_consumer_interface",
    "validate_consumer_screen_factory_config",
]
