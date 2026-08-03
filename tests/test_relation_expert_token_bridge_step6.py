from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    build_hlt_v3_cache,
    build_hlt_v3_cache_metadata,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (
    NativeHLTExpertDataset,
    NativeHLTExpertTrainingConfig,
    build_hlt_evidence_mode_contract,
    copy_offline_expert_initialization,
    infer_native_hlt_expert_replica,
    make_native_hlt_expert_loader,
    native_hlt_expert_objective,
    native_hlt_parameter_groups,
    train_native_hlt_expert,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_controls import (
    NativeHLTControlTrainingConfig,
    train_native_hlt_control,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (
    build_hlt_v3_profile_contract,
)
from teacher_logit_reco.relation_expert_token_bridge.native_fusion import (
    NativeFusionDataset,
    NativeFusionTrainingConfig,
    build_native_fusion_contract,
    build_native_fusion_model,
    evaluate_native_hlt_fusion,
    load_native_fusion_cache,
    publish_native_fusion_cache,
    train_native_hlt_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.step6 import (
    build_stage_d_run_registry,
    build_step6_bundle,
    execute_miniature_stage_d,
    materialize_stage_d_confirmation_rows,
    publish_step6_bundle,
    resolve_stage_d_run,
    validate_stage_d_run_registry,
    validate_stage_d_confirmation_registry,
    validate_step6_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _raw(events: int = 20, length: int = 8):
    rng = np.random.default_rng(6001)
    tokens = np.zeros((events, length, 14), dtype=np.float32)
    mask = np.ones((events, length), dtype=bool)
    tokens[:, :, 0] = rng.uniform(0.1, 40.0, size=(events, length))
    tokens[:, :, 1] = rng.uniform(-2.0, 2.0, size=(events, length))
    tokens[:, :, 2] = rng.uniform(-np.pi, np.pi, size=(events, length))
    tokens[:, :, 3] = (
        tokens[:, :, 0] * np.cosh(tokens[:, :, 1])
    ).astype(np.float32)
    tokens[:, :, 5] = 1.0
    tokens[:, :, 10:14] = rng.normal(
        0.0, 0.05, size=(events, length, 4)
    )
    identities = [f"jet-{index:04d}" for index in range(events)]
    return tokens, mask, identities


def _replica_caches(
    *, logical_role: str, policy: str
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict]]:
    tokens, mask, identities = _raw()
    profile = build_hlt_v3_profile_contract(
        raw_input_schema_sha256=SHA_A,
        hlt_replica_manifest_sha256=SHA_B,
    )
    replica_ids = (
        (0,)
        if logical_role != "model_train" or policy == "R_FIXED"
        else (0, 1, 2, 3)
    )
    arrays, metadata = {}, {}
    for replica in replica_ids:
        arrays[replica], diagnostics = build_hlt_v3_cache(
            tokens,
            mask,
            canonical_identities=identities,
            logical_role=logical_role,
            replica_id=replica,
            realization_policy=policy,
            profile_id="D_NOMINAL",
        )
        metadata[replica] = build_hlt_v3_cache_metadata(
            arrays=arrays[replica],
            diagnostics=diagnostics,
            logical_role=logical_role,
            replica_id=replica,
            realization_policy=policy,
            degradation_profile_id="D_NOMINAL",
            profile_contract=profile,
            split_manifest_sha256=SHA_A,
            identity_manifest_sha256=SHA_B,
            raw_input_sha256=SHA_C,
        )
    return arrays, metadata


def _native_dataset(role: str, policy: str | None = None):
    if policy is None:
        policy = (
            "R_MULTI"
            if role in {"model_train", "scale_train"}
            else "R_FIXED"
        )
    arrays, metadata = _replica_caches(logical_role=role, policy=policy)
    _, _, identities = _raw()
    return NativeHLTExpertDataset(
        replica_arrays=arrays,
        replica_metadata=metadata,
        labels=np.arange(20) % 10,
        identities=identities,
        logical_role=role,
        realization_policy=policy,
    )


def test_native_hlt_replica_cycle_and_evaluation_freeze() -> None:
    train = _native_dataset("model_train")
    seen = []
    for epoch in range(1, 5):
        train.set_epoch(epoch)
        seen.append(train[3]["replica_id"])
    assert sorted(seen) == [0, 1, 2, 3]
    validation = _native_dataset("val_stop")
    for epoch in (1, 2, 40):
        validation.set_epoch(epoch)
        assert validation[3]["replica_id"] == 0


def test_native_hlt_dataset_supports_authenticated_logical_subroles() -> None:
    arrays, metadata = _replica_caches(
        logical_role="val_design", policy="R_FIXED"
    )
    _, _, identities = _raw()
    selected = np.asarray([5, 2], dtype=np.int64)
    dataset = NativeHLTExpertDataset(
        replica_arrays=arrays,
        replica_metadata=metadata,
        labels=np.asarray([5, 2], dtype=np.int64),
        identities=[identities[index] for index in selected],
        logical_role="design_confirm",
        source_logical_role="val_design",
        realization_policy="R_FIXED",
        source_indices_by_replica={0: selected},
    )
    assert dataset[0]["identity"] == identities[5]
    assert np.array_equal(dataset[0]["tokens"], arrays[0]["tokens"][5])


def test_native_hlt_objectives_are_scientifically_separate() -> None:
    torch.manual_seed(6002)
    logits = torch.randn(4, 10, requires_grad=True)
    tokens = torch.randn(4, 2, 64, requires_grad=True)
    labels = torch.arange(4)
    loss, pieces = native_hlt_expert_objective(
        logits=logits,
        tokens=tokens,
        labels=labels,
        mode="HE_SCRATCH_CE",
    )
    loss.backward()
    assert pieces["token_alignment"].item() == 0.0
    assert logits.grad is not None
    with pytest.raises(ValueError, match="consumed offline targets"):
        native_hlt_expert_objective(
            logits=logits,
            tokens=tokens,
            labels=labels,
            mode="HE_OFFLINE_INIT",
            offline_target_tokens=tokens.detach(),
        )
    logits.grad = None
    tokens.grad = None
    loss, pieces = native_hlt_expert_objective(
        logits=logits,
        tokens=tokens,
        labels=labels,
        mode="HE_DUAL_OBJECTIVE",
        lambda_token=0.25,
        lambda_logit=0.50,
        offline_target_tokens=torch.zeros_like(tokens),
        offline_target_logits=torch.zeros_like(logits),
    )
    loss.backward()
    assert pieces["token_alignment"].item() > 0.0
    assert tokens.grad is not None


class _TinyParticle(torch.nn.Module):
    def __init__(self, measurement: bool):
        super().__init__()
        self.measurement_embedding_enabled = measurement
        self.projection = torch.nn.Linear(17, 64)
        self.measurement_state_embedding = (
            torch.nn.Embedding(3, 64) if measurement else None
        )


class _TinyNativeExpert(torch.nn.Module):
    def __init__(self, measurement: bool = False):
        super().__init__()
        self.particle_encoder = _TinyParticle(measurement)
        self.tokenizer = torch.nn.Linear(64, 128)
        self.classifier = torch.nn.Linear(64, 10)

    def forward(self, *, features, return_details=False, **_kwargs):
        pooled = features.mean(dim=2)
        hidden = torch.tanh(self.particle_encoder.projection(pooled))
        tokens = self.tokenizer(hidden).reshape(len(hidden), 2, 64)
        logits = self.classifier(tokens.mean(dim=1))
        return {"tokens": tokens, "logits": logits} if return_details else logits


class _TinyControl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(17, 10)

    def forward(self, *, features, **_kwargs):
        return self.projection(features.mean(dim=2))


def test_offline_initialization_and_two_learning_rate_groups() -> None:
    source = _TinyNativeExpert(False)
    target = _TinyNativeExpert(True)
    report = copy_offline_expert_initialization(target, source.state_dict())
    assert report["availability_embedding_zero_initialized"] is True
    assert torch.count_nonzero(
        target.particle_encoder.measurement_state_embedding.weight
    ).item() == 0
    groups = native_hlt_parameter_groups(
        target,
        mode="HE_OFFLINE_INIT",
        copied_parameter_names=report["copied_parameter_names"],
    )
    assert [group["lr"] for group in groups] == [1.0e-4, 5.0e-4]
    no_new = native_hlt_parameter_groups(
        source,
        mode="HE_OFFLINE_INIT",
        copied_parameter_names=list(dict(source.named_parameters())),
    )
    assert len(no_new) == 1 and no_new[0]["lr"] == 1.0e-4


def test_miniature_native_expert_runs_full_budget_and_reuses(tmp_path: Path) -> None:
    train = _native_dataset("model_train")
    val = _native_dataset("val_stop")
    train_loader = make_native_hlt_expert_loader(
        train, seed=101, training=True, batch_size=4
    )
    val_loader = make_native_hlt_expert_loader(
        val, seed=0, training=False, batch_size=5
    )
    config = NativeHLTExpertTrainingConfig(
        seed=101,
        mode="HE_SCRATCH_CE",
        maximum_epochs=2,
        microbatch_size=4,
        gradient_accumulation_steps=1,
        effective_batch_size=4,
        campaign_profile="miniature_test",
    )
    evidence = build_hlt_evidence_mode_contract()
    kwargs = dict(
        model=_TinyNativeExpert(),
        train_loader=train_loader,
        val_stop_loader=val_loader,
        output_dir=tmp_path / "expert",
        run_id="D_HLT_EXPERT_TEST",
        run_registry_sha256=SHA_A,
        lineage_hashes={"shared_hlt_normalizer": SHA_B},
        global_determinism_sha256=SHA_C,
        evidence_mode_contract_sha256=evidence["content_hash"],
        config=config,
    )
    first = train_native_hlt_expert(**kwargs)
    second = train_native_hlt_expert(**kwargs)
    assert second == first
    assert first["epochs_completed"] == 2
    assert first["fixed_epoch_budget_completed"] is True
    assert first["performance_based_termination"] is False
    assert first["ordinary_hlt_only_baseline"] is True
    assert first["training_realization_policy"] == "R_MULTI"
    assert first["evaluation_realization_policy"] == "R_FIXED"
    output = infer_native_hlt_expert_replica(
        model=kwargs["model"],
        dataset=train,
        replica_id=3,
        batch_size=6,
    )
    assert output["tokens"].shape == (20, 2, 64)
    assert output["logits"].shape == (20, 10)
    assert output["identities"].tolist() == list(train.identities)


def test_miniature_matched_control_runs_full_budget(tmp_path: Path) -> None:
    train = _native_dataset("model_train")
    val = _native_dataset("val_stop")
    train_loader = make_native_hlt_expert_loader(
        train, seed=101, training=True, batch_size=4
    )
    val_loader = make_native_hlt_expert_loader(
        val, seed=0, training=False, batch_size=5
    )
    registration = train_native_hlt_control(
        model=_TinyControl(),
        train_loader=train_loader,
        val_stop_loader=val_loader,
        output_dir=tmp_path / "control",
        run_id="D_H_BASE_TEST",
        run_registry_sha256=SHA_A,
        lineage_hashes={"shared_hlt_normalizer": SHA_B},
        global_determinism_sha256=SHA_C,
        config=NativeHLTControlTrainingConfig(
            seed=101,
            control_id="H_BASE",
            maximum_epochs=2,
            microbatch_size=4,
            gradient_accumulation_steps=1,
            effective_batch_size=4,
            campaign_profile="miniature_test",
        ),
    )
    assert registration["training_realization_policy"] == "R_MULTI"
    assert registration["evaluation_realization_policy"] == "R_FIXED"
    assert registration["epochs_completed"] == 2
    assert registration["offline_targets_consumed"] is False
    assert registration["performance_based_termination"] is False


def _fusion_cache(root: Path, split: str):
    rng = np.random.default_rng(6003)
    replicas = (0, 1, 2, 3) if split == "model_train" else (0,)
    tokens = {
        replica: {
            expert: rng.normal(size=(20, 2, 64)).astype(np.float32)
            for expert in EXPERT_ORDER
        }
        for replica in replicas
    }
    logits = {
        replica: {
            expert: rng.normal(size=(20, 10)).astype(np.float32)
            for expert in EXPERT_ORDER
        }
        for replica in replicas
    }
    return publish_native_fusion_cache(
        output_dir=root,
        split=split,
        pipeline_seed=101,
        shape_id="TEST_S2_64",
        realization_policy="R_MULTI",
        identities=[f"jet-{index:04d}" for index in range(20)],
        labels=np.arange(20) % 10,
        token_banks_by_replica=tokens,
        expert_logits_by_replica=logits,
        expert_registration_hashes={name: SHA_A for name in EXPERT_ORDER},
        hlt_cache_hashes_by_replica={replica: SHA_B for replica in replicas},
        identity_manifest_sha256=SHA_C,
        label_manifest_sha256=SHA_D,
    )


def test_native_fusions_cache_train_and_parameter_free_controls(
    tmp_path: Path,
) -> None:
    train = _fusion_cache(tmp_path / "train", "model_train")
    val = _fusion_cache(tmp_path / "val", "val_stop")
    train_path = tmp_path / "train" / "model_train_native_hlt_tokens.json"
    val_path = tmp_path / "val" / "val_stop_native_hlt_tokens.json"
    metadata, arrays = load_native_fusion_cache(train_path)
    dataset = NativeFusionDataset(metadata, arrays)
    dataset.set_epoch(1)
    replicas = [dataset[index]["replica_id"] for index in range(len(dataset))]
    assert set(replicas).issubset({0, 1, 2, 3})
    batch_tokens = {
        name: torch.from_numpy(arrays[f"tokens_r0_{name}"][:3])
        for name in EXPERT_ORDER
    }
    batch_logits = {
        name: torch.from_numpy(arrays[f"logits_r0_{name}"][:3])
        for name in EXPERT_ORDER
    }
    for variant in (
        "HF_NATIVE",
        "HF_LOGIT_MEAN",
        "HF_TRAINED_LOGIT",
        "HF_7X_UNBIASED_LOGIT_MEAN",
        "HF_7X_UNBIASED_TOKEN_FUSION",
    ):
        model = build_native_fusion_model(
            variant, bank_dimensions={name: 64 for name in EXPERT_ORDER}
        )
        assert model(
            token_banks=batch_tokens, expert_logits=batch_logits
        ).shape == (3, 10)
    mean_metrics = evaluate_native_hlt_fusion(
        model=build_native_fusion_model(
            "HF_LOGIT_MEAN",
            bank_dimensions={name: 64 for name in EXPERT_ORDER},
        ),
        manifest_path=val_path,
        batch_size=5,
    )
    assert mean_metrics["event_count"] == 20
    config = NativeFusionTrainingConfig(
        seed=101,
        variant="HF_TRAINED_LOGIT",
        maximum_epochs=2,
        batch_size=5,
        campaign_profile="miniature_test",
    )
    kwargs = dict(
        model=build_native_fusion_model(
            "HF_TRAINED_LOGIT",
            bank_dimensions={name: 64 for name in EXPERT_ORDER},
        ),
        model_train_manifest=train_path,
        val_stop_manifest=val_path,
        output_dir=tmp_path / "fusion_run",
        run_id="D_NATIVE_HLT_FUSION_TEST",
        run_registry_sha256=SHA_A,
        native_fusion_contract_sha256=build_native_fusion_contract()[
            "content_hash"
        ],
        global_determinism_sha256=SHA_B,
        config=config,
    )
    first = train_native_hlt_fusion(**kwargs)
    second = train_native_hlt_fusion(**kwargs)
    assert second == first
    assert first["model_train_cache_sha256"] == train["content_hash"]
    assert first["val_stop_cache_sha256"] == val["content_hash"]
    assert first["epochs_completed"] == 2
    assert first["offline_targets_consumed"] is False


def test_stage_d_registry_bundle_and_all_negative_miniature(
    tmp_path: Path,
) -> None:
    registry = build_stage_d_run_registry()
    assert validate_stage_d_run_registry(registry) == registry["content_hash"]
    assert registry["row_counts"] == {
        "scratch_expert_memberships": 42,
        "encoder_screen_memberships": 420,
        "bridge_parent_expert_memberships": 105,
        "native_fusion_memberships": 30,
        "matched_baselines": 4,
    }
    resolved = resolve_stage_d_run(
        registry, run_id=registry["scratch_expert_rows"][0]["run_id"]
    )
    assert resolved["configuration"]["mode"] == "HE_SCRATCH_CE"
    selected = [
        row["run_id"]
        for row in registry["scratch_expert_rows"]
        if row["configuration"]["shape_id"] == "S1_128"
    ] + [
        row["run_id"]
        for row in registry["native_fusion_rows"]
        if row["configuration"]["shape_id"] == "S1_128"
        and row["configuration"]["fusion_variant"]
        in {"HF_NATIVE", "HF_TRAINED_LOGIT"}
    ]
    confirmation = materialize_stage_d_confirmation_rows(
        registry, selected_run_ids=selected
    )
    assert (
        validate_stage_d_confirmation_registry(confirmation)
        == confirmation["content_hash"]
    )
    assert len(confirmation["rows"]) == 33
    assert {row["seed"] for row in confirmation["rows"]} == {101, 202, 303}

    def completed(_row):
        return {
            "status": "completed",
            "accuracy": 0.0,
            "performance_based_termination": False,
        }

    completion = execute_miniature_stage_d(
        registry,
        expert_executor=completed,
        fusion_executor=completed,
        baseline_executor=completed,
    )
    assert completion["native_specialization_measurable"] is True
    snapshot = source_snapshot(Path(__file__).resolve().parents[1])
    bundle = build_step6_bundle(
        campaign_spec_sha256=SHA_A,
        step5_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        hlt_replica_manifest_sha256=SHA_D,
        shared_hlt_normalizer_sha256="e" * 64,
        source_snapshot=snapshot,
    )
    assert validate_step6_bundle(bundle) == bundle["step6_bundle"]["content_hash"]
    publication = publish_step6_bundle(
        campaign_root=tmp_path / "campaign", bundle=bundle
    )
    assert publication["step6_bundle_sha256"] == bundle["step6_bundle"][
        "content_hash"
    ]


def test_step6_production_workers_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    expert = (
        root / "sbatch" / "run_retb_train_native_hlt_expert.sh"
    ).read_text(encoding="utf-8")
    fusion = (
        root / "sbatch" / "run_retb_train_native_hlt_fusion.sh"
    ).read_text(encoding="utf-8")
    cache = (
        root / "sbatch" / "run_retb_build_native_hlt_fusion_cache.sh"
    ).read_text(encoding="utf-8")
    control = (
        root / "sbatch" / "run_retb_train_native_hlt_control.sh"
    ).read_text(encoding="utf-8")
    assert "--offline-checkpoint" in expert
    assert "--train-cache" in expert and "--val-stop-cache" in expert
    assert "--model-train-cache" in fusion and "--val-stop-cache" in fusion
    assert "--expert-output-manifest" in cache
    assert "--wide-capacity-artifact" not in control
    for worker in (expert, fusion, cache, control):
        assert "PYTHONNOUSERSITE=1" in worker
        assert "PYTHONDONTWRITEBYTECODE=1" in worker
