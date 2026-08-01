#!/usr/bin/env python3
"""Execute and authenticate every declared RETB Stage-C offline capacity control."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_offline_expert import (  # noqa: E402
    _dataset,
    _load_npz,
    _load_trees,
)
from teacher_logit_reco.relation_expert_token_bridge.capacity import (  # noqa: E402
    OFFLINE_CAPACITY_CONTROL_ORDER,
    build_offline_capacity_control_registration,
    build_offline_capacity_execution_registry,
    build_offline_long_exposure_ledger,
    select_monolithic_capacity_controls,
    validate_offline_capacity_control_registration,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (  # noqa: E402
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    _file_sha256,
    make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    TokenTransformerFusion,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    FAMILIES,
    GroupedHeadRelationParticleTransformer,
    MonolithicBase4ParticleTransformer,
    OfflineClassifierAdapter,
    analytical_particle_transformer_flops,
    build_monolithic_grid,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (  # noqa: E402
    OfflineCapacityTrainingConfig,
    _collect_predictions,
    build_capacity_profile,
    build_capacity_val_design_metrics,
    train_offline_capacity_model,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.capacity import (  # noqa: E402
    pair_encoder_flops,
    select_wide_widths,
)
from teacher_logit_reco.relational_part.model import (  # noqa: E402
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
    WideBaseParticleTransformer,
)

import torch  # noqa: E402


SEVEN_SEEDS = (101, 202, 303, 404, 505, 606, 707)
BASE_CONFIG = (128, 4, 8, 8, 2)
LOCKED_OFFLINE_SHAPES_RELATIVE_PATH = Path(
    "selection/stage_c/locked_offline_shapes.json"
)


def _locked_offline_shapes_path(root: Path) -> Path:
    return root / LOCKED_OFFLINE_SHAPES_RELATIVE_PATH


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("capacity checkpoint lacks a model state dictionary")
    return state


def _state_parameter_count(path: Path) -> int:
    return sum(
        int(value.numel())
        for value in _state(path).values()
        if isinstance(value, torch.Tensor)
    )


def _loaders(
    root: Path,
    *,
    batch_size: int,
    seed: int,
    training_role: str = "model_train",
) -> tuple[Any, Any, Any]:
    datasets = {}
    for split in (training_role, "val_stop", "val_design"):
        path = (
            root
            / "inputs"
            / "offline"
            / split
            / "offline_inputs.npz"
        )
        arrays = _load_npz(path)
        identities = [str(value) for value in arrays["identities"].tolist()]
        trees = _load_trees(
            root / "inputs" / "region_tree" / "offline",
            split=split,
            identities=identities,
        )
        datasets[split] = _dataset(arrays, region_trees=trees)
    return (
        make_offline_expert_loader(
            datasets[training_role],
            seed=seed,
            training=True,
            batch_size=batch_size,
        ),
        make_offline_expert_loader(
            datasets["val_stop"],
            seed=seed,
            training=False,
            batch_size=batch_size,
        ),
        make_offline_expert_loader(
            datasets["val_design"],
            seed=seed,
            training=False,
            batch_size=batch_size,
        ),
    )


def _lineage(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    training_role: str = "model_train",
    normalization_population: str = "500k",
) -> dict[str, str]:
    values = {
        "campaign_spec": campaign["content_hash"],
        "stage_c_registry": load_hashed_json(
            root / "registry" / "retb_stage_c_runs.json"
        )["content_hash"],
        "shape_high": load_hashed_json(
            _locked_offline_shapes_path(root)
        )["content_hash"],
        "relation_normalization": load_hashed_json(
            root
            / "inputs"
            / "normalization"
            / f"offline_{normalization_population}"
            / "relation.json"
        )["content_hash"],
        "region_normalization": load_hashed_json(
            root
            / "inputs"
            / "normalization"
            / f"offline_{normalization_population}"
            / "region.json"
        )["content_hash"],
    }
    for split in (training_role, "val_stop", "val_design"):
        values[f"{split}_offline_input"] = load_hashed_json(
            root
            / "inputs"
            / "offline"
            / split
            / "offline_input_manifest.json"
        )["content_hash"]
    return values


def _ledger(
    *,
    root: Path,
    control_id: str,
    presentations: int,
    training_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = bind_source(
        with_content_hash(
            {
                "contract": "retb_offline_capacity_label_exposure_v1",
                "schema_version": 1,
                "control_id": control_id,
                "labeled_example_presentations": int(presentations),
                "component_training_hashes": [training_sha256],
                "counting_unit": (
                    "one labeled event presented in one optimizer phase"
                ),
                "repeated_presentations_count_repeatedly": True,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root
        / "runs"
        / "stage_c"
        / "capacity_controls"
        / control_id
        / "label_exposure.json",
        artifact,
    )
    return artifact


def _finalize_training(
    *,
    root: Path,
    control_id: str,
    training: Mapping[str, Any],
    profile: Mapping[str, Any],
    execution_sha: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    output = (
        root / "runs" / "stage_c" / "capacity_controls" / control_id
    )
    registration_path = output / "control_registration.json"
    if registration_path.is_file():
        existing = load_hashed_json(registration_path)
        validate_offline_capacity_control_registration(existing)
        if existing.get("control_id") != control_id:
            raise ValueError("reusable capacity control ID differs")
        return existing
    ledger = _ledger(
        root=root,
        control_id=control_id,
        presentations=int(training["labeled_example_presentations"]),
        training_sha256=training["content_hash"],
        source=source,
    )
    registration = bind_source(
        build_offline_capacity_control_registration(
            control_id=control_id,
            execution_registry_sha256=execution_sha,
            checkpoint_hashes=[training["checkpoint_sha256"]],
            parameter_count=int(profile["parameter_count"]),
            inference_flops_batch1=int(profile["inference_flops_batch1"]),
            inference_flops_batch128=int(
                profile["inference_flops_batch128"]
            ),
            profile_sha256=profile["content_hash"],
            labeled_example_presentations=int(
                training["labeled_example_presentations"]
            ),
            label_exposure_ledger_sha256=ledger["content_hash"],
            training_artifact_hashes=[training["content_hash"]],
            val_design_prediction_sha256=training[
                "val_design_prediction_sha256"
            ],
            val_design_metrics_sha256=training[
                "val_design_metrics_sha256"
            ],
            fixed_budget_completed=True,
        ),
        source_snapshot=source,
    )
    write_immutable_json(registration_path, registration)
    return registration


def _train(
    *,
    root: Path,
    campaign: Mapping[str, Any],
    control_id: str,
    model: Any,
    flops: int,
    execution_sha: str,
    lineage: Mapping[str, str],
    source: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int = 101,
    exact_updates: int | None = None,
    output_suffix: str | None = None,
    training_role: str = "model_train",
    output_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    miniature = campaign["campaign_profile"] == "miniature_test"
    train_loader, val_stop_loader, val_design_loader = _loaders(
        root,
        batch_size=batch_size,
        seed=seed,
        training_role=training_role,
    )
    output = (
        (
            root
            / "runs"
            / "stage_c"
            / "capacity_controls"
            / control_id
        )
        if output_root is None
        else output_root / control_id
    ) / (output_suffix or "training")
    profile = build_capacity_profile(
        control_id=control_id,
        model=model,
        analytical_flops_batch1=int(flops),
        source_snapshot=source,
    )
    profile_path = output / "complete_graph_profile.json"
    registration_path = output / "training_registration.json"
    if registration_path.is_file():
        existing_profile = load_hashed_json(profile_path)
        existing = load_hashed_json(
            registration_path,
            expected_contract=(
                "retb_offline_capacity_training_registration_v1"
            ),
        )
        if (
            existing_profile != profile
            or existing.get("control_id") != control_id
            or existing.get("profile_sha256") != profile["content_hash"]
            or existing.get("fixed_budget_completed") is not True
        ):
            raise ValueError("reusable capacity training lineage differs")
        return existing, existing_profile
    write_immutable_json(profile_path, profile)
    training = train_offline_capacity_model(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_stop_loader,
        val_design_loader=val_design_loader,
        output_dir=output,
        config=OfflineCapacityTrainingConfig(
            control_id=control_id,
            seed=seed,
            maximum_epochs=2 if miniature else 40,
            microbatch_size=batch_size,
            gradient_accumulation_steps=2,
            campaign_profile=(
                "miniature_test" if miniature else "production"
            ),
            exact_optimizer_update_budget=exact_updates,
        ),
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        execution_registry_sha256=execution_sha,
        lineage_hashes=lineage,
        profile=profile,
        source_snapshot=source,
        device=device,
    )
    return training, profile


class _MeanLogitGraph(torch.nn.Module):
    def __init__(self, members: Sequence[Any]) -> None:
        super().__init__()
        self.members = torch.nn.ModuleList(list(members))

    def forward(self, **batch: Any) -> Any:
        return torch.stack(
            [member(**batch) for member in self.members], dim=0
        ).mean(dim=0)


class _SevenTokenGraph(torch.nn.Module):
    def __init__(
        self,
        members: Sequence[RetbExpertModel],
        *,
        token_dimension: int,
    ) -> None:
        super().__init__()
        if len(members) != 7:
            raise ValueError("seven-token control requires seven experts")
        self.members = torch.nn.ModuleList(list(members))
        self.fusion = TokenTransformerFusion(
            bank_dimensions={
                name: int(token_dimension)
                for name in (
                    "BASE4",
                    "PT",
                    "TRACK",
                    "PID",
                    "CHARGE",
                    "DENSITY",
                    "REGION",
                )
            }
        )

    def forward(self, **batch: Any) -> Any:
        names = (
            "BASE4",
            "PT",
            "TRACK",
            "PID",
            "CHARGE",
            "DENSITY",
            "REGION",
        )
        banks = {}
        for name, member in zip(names, self.members):
            details = member(return_details=True, **batch)
            banks[name] = details["tokens"]
        return self.fusion(token_banks=banks)


def _attest_composite(
    *,
    root: Path,
    campaign: Mapping[str, Any],
    control_id: str,
    model: Any,
    member_registrations: Sequence[Mapping[str, Any]],
    flops: int,
    execution_sha: str,
    source: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    output = root / "runs" / "stage_c" / "capacity_controls" / control_id
    output.mkdir(parents=True, exist_ok=True)
    registration_path = output / "control_registration.json"
    if registration_path.is_file():
        existing = load_hashed_json(registration_path)
        validate_offline_capacity_control_registration(existing)
        if existing.get("control_id") != control_id:
            raise ValueError("reusable composite capacity control differs")
        return existing
    profile = build_capacity_profile(
        control_id=control_id,
        model=model,
        analytical_flops_batch1=flops,
        source_snapshot=source,
    )
    write_immutable_json(output / "complete_graph_profile.json", profile)
    _, _, loader = _loaders(root, batch_size=batch_size, seed=101)
    prediction = _collect_predictions(model.to(device), loader, device=device)
    prediction_path = output / "val_design_predictions.npz"
    with prediction_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            logits=prediction["logits"],
            labels=prediction["labels"],
            identities=np.asarray(prediction["identities"], dtype="U"),
        )
    metrics = bind_source(
        build_capacity_val_design_metrics(
            classification_metrics=prediction["metrics"],
            control_id=control_id,
            checkpoint_sha256=None,
        ),
        source_snapshot=source,
    )
    write_immutable_json(output / "val_design_metrics.json", metrics)
    presentations = sum(
        int(row["labeled_example_presentations"])
        for row in member_registrations
    )
    ledger = _ledger(
        root=root,
        control_id=control_id,
        presentations=presentations,
        training_sha256=member_registrations[0]["content_hash"],
        source=source,
    )
    registration = bind_source(
        build_offline_capacity_control_registration(
            control_id=control_id,
            execution_registry_sha256=execution_sha,
            checkpoint_hashes=[
                row["checkpoint_sha256"] for row in member_registrations
            ],
            parameter_count=profile["parameter_count"],
            inference_flops_batch1=profile["inference_flops_batch1"],
            inference_flops_batch128=profile["inference_flops_batch128"],
            profile_sha256=profile["content_hash"],
            labeled_example_presentations=presentations,
            label_exposure_ledger_sha256=ledger["content_hash"],
            training_artifact_hashes=[
                row["content_hash"] for row in member_registrations
            ],
            val_design_prediction_sha256=_sha(prediction_path),
            val_design_metrics_sha256=metrics["content_hash"],
            fixed_budget_completed=True,
        ),
        source_snapshot=source,
    )
    write_immutable_json(registration_path, registration)
    return registration


def _expert_model(
    *,
    shape_id: str,
    normalization: Mapping[str, Any],
    region_normalization: Mapping[str, Any],
    weaver: Any,
) -> RetbExpertModel:
    return RetbExpertModel(
        particle_encoder=RetbParticleEncoder(
            expert_id="BASE4",
            topology="B_CONCAT",
            weaver_module=weaver,
            normalization_artifact=normalization,
            region_normalization_artifact=region_normalization,
            activation_checkpointing=True,
        ),
        shape_id=shape_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    for role in ("model_train", "val_stop", "val_design"):
        authorize_dataset_access(
            worker_role=(
                "training_worker" if role != "val_design" else "design_worker"
            ),
            requested_resource=role,
        )
    source = source_snapshot(REPO_ROOT)
    execution = load_hashed_json(
        root / "registry" / "retb_offline_capacity_execution.json"
    )
    expected_execution = build_offline_capacity_execution_registry()
    actual = dict(execution)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected_execution.pop("content_hash", None)
    if actual != expected_execution:
        raise ValueError("offline capacity execution registry differs")
    execution_sha = execution["content_hash"]
    lineage = _lineage(root, campaign)
    normalization = load_hashed_json(
        root
        / "inputs"
        / "normalization"
        / "offline_500k"
        / "relation.json"
    )
    region = load_hashed_json(
        root
        / "inputs"
        / "normalization"
        / "offline_500k"
        / "region.json"
    )
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    registrations: dict[str, dict[str, Any]] = {}

    raw_models = {
        "O_BASE": OfflineClassifierAdapter(
            RelationalParticleTransformer(weaver_module=weaver)
        ),
        "O_WIDE": OfflineClassifierAdapter(
            WideBaseParticleTransformer(
                capacity_artifact=select_wide_widths(),
                weaver_module=weaver,
            )
        ),
        "O_FULLREL": OfflineClassifierAdapter(
            RelationalFamilyParticleTransformer(
                FAMILIES,
                normalization_artifact=normalization,
                region_normalization_artifact=region,
                weaver_module=weaver,
            )
        ),
        "O_GROUPED_HEAD_REL": OfflineClassifierAdapter(
            GroupedHeadRelationParticleTransformer(
                normalization_artifact=normalization,
                region_normalization_artifact=region,
                weaver_module=weaver,
            )
        ),
    }
    base_flops = analytical_particle_transformer_flops(
        configuration=BASE_CONFIG
    )
    raw_flops = {
        "O_BASE": base_flops + pair_encoder_flops(4, (64, 64, 64)),
        "O_WIDE": base_flops
        + pair_encoder_flops(
            4, tuple(select_wide_widths()["selected_widths"])
        ),
        "O_FULLREL": base_flops + pair_encoder_flops(62, (64, 64, 64)),
        "O_GROUPED_HEAD_REL": base_flops
        + pair_encoder_flops(4, (64, 64, 64))
        + sum(
            pair_encoder_flops(
                int(
                    __import__(
                        "teacher_logit_reco.relational_part.pair_builder",
                        fromlist=["SUPPORTED_FAMILY_DIMENSIONS"],
                    ).SUPPORTED_FAMILY_DIMENSIONS[name]
                ),
                (64, 64, 64),
            )
            for name in FAMILIES
        ),
    }
    for control_id, model in raw_models.items():
        training, profile = _train(
            root=root,
            campaign=campaign,
            control_id=control_id,
            model=model,
            flops=raw_flops[control_id],
            execution_sha=execution_sha,
            lineage=lineage,
            source=source,
            batch_size=args.batch_size,
            device=device,
        )
        registrations[control_id] = _finalize_training(
            root=root,
            control_id=control_id,
            training=training,
            profile=profile,
            execution_sha=execution_sha,
            source=source,
        )

    # Resolve the complete selected relation-token graph before selecting the
    # two monolithic matches.  This prevents encoder-only matching.
    relation_registration = load_hashed_json(
        root
        / "selection"
        / "offline_fusions"
        / "SHAPE_HIGH"
        / "seed_101"
        / "fusion_registration.json"
    )
    expert_regs = [
        load_hashed_json(
            root
            / "selection"
            / "offline_experts"
            / "SHAPE_HIGH"
            / expert
            / "seed_101"
            / "checkpoint_registration.json"
        )
        for expert in (
            "BASE4",
            "PT",
            "TRACK",
            "PID",
            "CHARGE",
            "DENSITY",
            "REGION",
        )
    ]
    model_train_events = int(
        load_hashed_json(
            root
            / "inputs"
            / "offline"
            / "model_train"
            / "offline_input_manifest.json"
        )["event_count"]
    )
    expert_presentations = [
        model_train_events * int(row["epochs_completed"])
        for row in expert_regs
    ]
    relation_presentations = (
        model_train_events * int(relation_registration["epochs_completed"])
    )
    relation_checkpoint_path = (
        root
        / "selection"
        / "offline_fusions"
        / "SHAPE_HIGH"
        / "seed_101"
        / "best_model_val.pt"
    )
    target_parameters = int(
        sum(int(row["trainable_parameter_count"]) for row in expert_regs)
        + _state_parameter_count(relation_checkpoint_path)
    )
    target_flops = sum(raw_flops["O_BASE"] for _ in range(7)) + base_flops
    candidates = []
    for configuration in build_monolithic_grid():
        model = MonolithicBase4ParticleTransformer(
            configuration, weaver_module=weaver
        )
        candidates.append(
            {
                "configuration": list(configuration),
                "parameter_count": sum(
                    int(value.numel()) for value in model.parameters()
                ),
                "inference_flops_batch1": (
                    analytical_particle_transformer_flops(
                        configuration=configuration
                    )
                    + pair_encoder_flops(4, (64, 64, 64))
                ),
                "inference_flops_batch128": 128
                * (
                    analytical_particle_transformer_flops(
                        configuration=configuration
                    )
                    + pair_encoder_flops(4, (64, 64, 64))
                ),
            }
        )
        del model
    mono_selection = bind_source(
        select_monolithic_capacity_controls(
            target_parameters=target_parameters,
            target_flops_batch1=target_flops,
            target_flops_batch128=128 * target_flops,
            candidates=candidates,
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root / "selection" / "offline_monolithic_capacity.json",
        mono_selection,
    )
    for control_id in ("O_MONO_PARAM", "O_MONO_FLOP"):
        configuration = mono_selection[control_id]["configuration"]
        model = OfflineClassifierAdapter(
            MonolithicBase4ParticleTransformer(
                configuration, weaver_module=weaver
            )
        )
        flops = int(mono_selection[control_id]["inference_flops_batch1"])
        training, profile = _train(
            root=root,
            campaign=campaign,
            control_id=control_id,
            model=model,
            flops=flops,
            execution_sha=execution_sha,
            lineage={**lineage, "monolithic_selection": mono_selection["content_hash"]},
            source=source,
            batch_size=args.batch_size,
            device=device,
        )
        registrations[control_id] = _finalize_training(
            root=root,
            control_id=control_id,
            training=training,
            profile=profile,
            execution_sha=execution_sha,
            source=source,
        )

    # The long baseline budget is serialized before training and remains fixed
    # even if every preceding control underperforms.
    long_ledger = bind_source(
        build_offline_long_exposure_ledger(
            component_rows=[
                {
                    "component_id": f"selected_expert_{index}",
                    "component_kind": "offline_expert",
                    "labeled_example_presentations": int(presentations),
                    "parent_sha256": row["content_hash"],
                }
                for index, (row, presentations) in enumerate(
                    zip(expert_regs, expert_presentations)
                )
            ]
            + [
                {
                    "component_id": "selected_primary_fusion",
                    "component_kind": "primary_frozen_fusion",
                    "labeled_example_presentations": relation_presentations,
                    "parent_sha256": relation_registration["content_hash"],
                }
            ],
            obase_effective_batch_size=128,
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root / "selection" / "offline_long_exposure_ledger.json",
        long_ledger,
    )
    long_model = OfflineClassifierAdapter(
        RelationalParticleTransformer(weaver_module=weaver)
    )
    training, profile = _train(
        root=root,
        campaign=campaign,
        control_id="O_BASE_LONG",
        model=long_model,
        flops=raw_flops["O_BASE"],
        execution_sha=execution_sha,
        lineage={
            **lineage,
            "long_exposure_ledger": long_ledger["content_hash"],
        },
        source=source,
        batch_size=args.batch_size,
        device=device,
        exact_updates=int(long_ledger["optimizer_update_budget"]),
    )
    registrations["O_BASE_LONG"] = _finalize_training(
        root=root,
        control_id="O_BASE_LONG",
        training=training,
        profile=profile,
        execution_sha=execution_sha,
        source=source,
    )

    # Seven independent ordinary BASE4 members, followed by a parameter-free
    # mean-logit combiner.
    ensemble_members, ensemble_rows = [], []
    for seed in SEVEN_SEEDS:
        member = OfflineClassifierAdapter(
            RelationalParticleTransformer(weaver_module=weaver)
        )
        training, _ = _train(
            root=root,
            campaign=campaign,
            control_id="O_7X_UNBIASED_ENSEMBLE",
            model=member,
            flops=raw_flops["O_BASE"],
            execution_sha=execution_sha,
            lineage=lineage,
            source=source,
            batch_size=args.batch_size,
            device=device,
            seed=seed,
            output_suffix=f"member_seed_{seed}",
        )
        member.load_state_dict(
            _state(
                root
                / "runs"
                / "stage_c"
                / "capacity_controls"
                / "O_7X_UNBIASED_ENSEMBLE"
                / f"member_seed_{seed}"
                / "best_model_val.pt"
            ),
            strict=True,
        )
        ensemble_members.append(member)
        ensemble_rows.append(training)
    registrations["O_7X_UNBIASED_ENSEMBLE"] = _attest_composite(
        root=root,
        campaign=campaign,
        control_id="O_7X_UNBIASED_ENSEMBLE",
        model=_MeanLogitGraph(ensemble_members),
        member_registrations=ensemble_rows,
        flops=7 * raw_flops["O_BASE"],
        execution_sha=execution_sha,
        source=source,
        batch_size=args.batch_size,
        device=device,
    )

    # Seven independent token experts followed by a fresh token transformer.
    locked_shapes = load_hashed_json(
        _locked_offline_shapes_path(root)
    )
    shape_id = str(locked_shapes["SHAPE_HIGH"]["shape_id"])
    token_members, token_rows = [], []
    for seed in SEVEN_SEEDS:
        member = _expert_model(
            shape_id=shape_id,
            normalization=normalization,
            region_normalization=region,
            weaver=weaver,
        )
        training, _ = _train(
            root=root,
            campaign=campaign,
            control_id="O_7X_UNBIASED_TOKEN_FUSION",
            model=member,
            flops=raw_flops["O_BASE"],
            execution_sha=execution_sha,
            lineage=lineage,
            source=source,
            batch_size=args.batch_size,
            device=device,
            seed=seed,
            output_suffix=f"member_seed_{seed}",
        )
        member.load_state_dict(
            _state(
                root
                / "runs"
                / "stage_c"
                / "capacity_controls"
                / "O_7X_UNBIASED_TOKEN_FUSION"
                / f"member_seed_{seed}"
                / "best_model_val.pt"
            ),
            strict=True,
        )
        for parameter in member.parameters():
            parameter.requires_grad_(False)
        token_members.append(member)
        token_rows.append(training)
    token_dimension = int(token_members[0].token_dimension)
    token_graph = _SevenTokenGraph(
        token_members, token_dimension=token_dimension
    )
    fusion_training, fusion_profile = _train(
        root=root,
        campaign=campaign,
        control_id="O_7X_UNBIASED_TOKEN_FUSION",
        model=token_graph,
        flops=7 * raw_flops["O_BASE"] + base_flops,
        execution_sha=execution_sha,
        lineage={
            **lineage,
            **{
                f"member_seed_{seed}": row["content_hash"]
                for seed, row in zip(SEVEN_SEEDS, token_rows)
            },
        },
        source=source,
        batch_size=args.batch_size,
        device=device,
        output_suffix="fusion_training",
    )
    # Count all member label exposure plus the fresh fusion phase in a
    # separately persisted composite lineage artifact.
    combined_training = bind_source(
        with_content_hash(
            {
                "contract": (
                    "retb_offline_capacity_composite_training_lineage_v1"
                ),
                "schema_version": 1,
                "control_id": "O_7X_UNBIASED_TOKEN_FUSION",
                "member_training_hashes": [
                    row["content_hash"] for row in token_rows
                ],
                "fusion_training_sha256": fusion_training["content_hash"],
                "checkpoint_sha256": fusion_training["checkpoint_sha256"],
                "val_design_prediction_sha256": fusion_training[
                    "val_design_prediction_sha256"
                ],
                "val_design_metrics_sha256": fusion_training[
                    "val_design_metrics_sha256"
                ],
                "labeled_example_presentations": int(
                    fusion_training["labeled_example_presentations"]
                    + sum(
                        row["labeled_example_presentations"]
                        for row in token_rows
                    )
                ),
                "fixed_budget_completed": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root
        / "runs"
        / "stage_c"
        / "capacity_controls"
        / "O_7X_UNBIASED_TOKEN_FUSION"
        / "composite_training_lineage.json",
        combined_training,
    )
    registrations["O_7X_UNBIASED_TOKEN_FUSION"] = _finalize_training(
        root=root,
        control_id="O_7X_UNBIASED_TOKEN_FUSION",
        training=combined_training,
        profile=fusion_profile,
        execution_sha=execution_sha,
        source=source,
    )

    # The selected relation-token graph has already completed its seven expert
    # and fusion training phases.  Its existing design inference is reused
    # only after every component hash is authenticated.
    relation_inference = load_hashed_json(
        root
        / "selection"
        / "offline_fusions"
        / "SHAPE_HIGH"
        / "seed_101"
        / "val_design_inference.json"
    )
    relation_output = (
        root
        / "runs"
        / "stage_c"
        / "capacity_controls"
        / "O_RELATION_EXPERT_TOKEN_FUSION"
    )
    relation_output.mkdir(parents=True, exist_ok=True)
    relation_profile = bind_source(
        with_content_hash(
            {
                "contract": "retb_offline_capacity_profile_v1",
                "schema_version": 1,
                "control_id": "O_RELATION_EXPERT_TOKEN_FUSION",
                "parameter_count": target_parameters,
                "inference_flops_batch1": target_flops,
                "inference_flops_batch128": 128 * target_flops,
                "maximum_particles": 128,
                "multiply_add_counts_as_two_flops": True,
                "analytical_not_measured": True,
                "measured_latency_used_for_selection": False,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        relation_output / "complete_graph_profile.json", relation_profile
    )
    presentations = int(
        sum(expert_presentations) + relation_presentations
    )
    relation_ledger = _ledger(
        root=root,
        control_id="O_RELATION_EXPERT_TOKEN_FUSION",
        presentations=presentations,
        training_sha256=relation_registration["content_hash"],
        source=source,
    )
    relation_checkpoints = [
        row["checkpoint_sha256"] for row in expert_regs
    ] + [relation_registration["checkpoint_sha256"]]
    registrations["O_RELATION_EXPERT_TOKEN_FUSION"] = bind_source(
        build_offline_capacity_control_registration(
            control_id="O_RELATION_EXPERT_TOKEN_FUSION",
            execution_registry_sha256=execution_sha,
            checkpoint_hashes=relation_checkpoints,
            parameter_count=target_parameters,
            inference_flops_batch1=target_flops,
            inference_flops_batch128=128 * target_flops,
            profile_sha256=relation_profile["content_hash"],
            labeled_example_presentations=presentations,
            label_exposure_ledger_sha256=relation_ledger["content_hash"],
            training_artifact_hashes=[
                row["content_hash"] for row in expert_regs
            ]
            + [relation_registration["content_hash"]],
            val_design_prediction_sha256=relation_inference[
                "prediction_file_sha256"
            ],
            val_design_metrics_sha256=relation_inference["content_hash"],
            fixed_budget_completed=True,
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        relation_output / "control_registration.json",
        registrations["O_RELATION_EXPERT_TOKEN_FUSION"],
    )
    if set(registrations) != set(OFFLINE_CAPACITY_CONTROL_ORDER):
        raise RuntimeError("offline capacity control coverage differs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
