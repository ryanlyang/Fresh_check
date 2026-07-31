from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.capacity import (
    select_monolithic_capacity_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (
    CAPACITY_COMPONENTS,
    DeployableRetbGraph,
    JointBridgeDeployableFrontend,
    build_complete_graph_capacity,
    export_deployable_retb_graph,
    load_deployable_retb_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (
    FinalConsumerDataset,
    FinalConsumerTrainingConfig,
    evaluate_final_consumer,
    load_selected_final_consumer_checkpoint,
    make_final_consumer_loader,
    train_final_consumer,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumers import (
    ADAPTER_VARIANTS,
    NATIVE_DROPOUT_MODES,
    REFINER_VARIANTS,
    UNRESTRICTED_EVIDENCE_VARIANTS,
    HLTResidualAdapter,
    NativeConditionedTokenRefiner,
    UnrestrictedHLTFusion,
    confidence_corruption_probabilities,
    deterministic_robust_mixture,
    sample_native_dropout,
    select_matched_token_mlp_width,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.replicas import (
    replica_for,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (
    build_final_consumer_policy,
    build_final_consumer_registry,
    build_step12_bundle,
    materialize_final_consumer_run,
    validate_final_consumer_registry,
    validate_materialized_final_consumer_run,
    validate_step12_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}
DIMS = {expert: 64 for expert in EXPERT_ORDER}
COUNTS = {expert: 2 for expert in EXPERT_ORDER}
UNCERTAINTY = {expert: 1 for expert in EXPERT_ORDER}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)

    def forward(self, values):
        return self.linear(values.mean(dim=1))


class _Fusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7 * 64, 10)

    def forward(self, *, token_banks):
        return self.linear(
            torch.cat(
                [
                    token_banks[expert].mean(dim=1)
                    for expert in EXPERT_ORDER
                ],
                dim=-1,
            )
        )


def _evidence(batch: int = 4):
    torch.manual_seed(1201)
    predicted = {
        expert: torch.randn(batch, 2, 64) for expert in EXPERT_ORDER
    }
    native = {
        expert: torch.randn(batch, 2, 64) for expert in EXPERT_ORDER
    }
    uncertainty = {
        expert: torch.randn(batch, 2, 1).clamp(-2, 2)
        for expert in EXPERT_ORDER
    }
    logits = {
        expert: torch.randn(batch, 10) for expert in EXPERT_ORDER
    }
    return predicted, native, uncertainty, logits


def test_step12_policy_registry_and_bundle_are_complete() -> None:
    policy = build_final_consumer_policy()
    registry = build_final_consumer_registry(
        step11_bundle_sha256=SHA_A,
        predictor_bundle_lock_sha256=SHA_B,
        policy_sha256=policy["content_hash"],
    )
    assert validate_final_consumer_registry(registry) == registry[
        "content_hash"
    ]
    assert registry["membership_count"] == 108
    assert {row["consumer_kind"] for row in registry["rows"]} == {
        "PF_FROZEN",
        "OF_ROBUST",
        "TR_REFINE",
        "HF_ADAPTER",
        "HF_UNRESTRICTED",
    }
    bundle = build_step12_bundle(
        campaign_spec_sha256=SHA_A,
        step11_bundle_sha256=SHA_B,
        predictor_bundle_lock_sha256=SHA_C,
        joint_campaign_lock_sha256=_digest("joint-campaign-lock"),
        global_determinism_sha256=_digest("determinism"),
        source_snapshot=SOURCE,
    )
    assert validate_step12_bundle(bundle) == bundle["step12_bundle"][
        "content_hash"
    ]


def test_robust_mixture_is_exact_and_deterministic() -> None:
    identities = [f"event-{index}" for index in range(8)]
    first = deterministic_robust_mixture(
        identities=identities, zero_based_epoch=0
    )
    second = deterministic_robust_mixture(
        identities=identities, zero_based_epoch=0
    )
    assert first == second
    counts = {
        mode: sum(row["mode"] == mode for row in first)
        for mode in (
            "all_oracle",
            "exactly_one_predicted",
            "independent_p0.5",
            "all_predicted",
        )
    }
    assert set(counts.values()) == {2}
    assert all(
        len(row["predicted_experts"]) == 1
        for row in first
        if row["mode"] == "exactly_one_predicted"
    )


def test_native_dropout_modes_preserve_expected_semantics() -> None:
    _, _, uncertainty, _ = _evidence(batch=32)
    probabilities = confidence_corruption_probabilities(uncertainty)
    combined = torch.stack(
        [
            probabilities[source][expert]
            for source in ("predicted", "native")
            for expert in EXPERT_ORDER
        ],
        dim=1,
    )
    assert torch.allclose(
        combined.mean(dim=1),
        torch.full((32,), 0.10),
        atol=2.0e-6,
        rtol=0,
    )
    for mode in NATIVE_DROPOUT_MODES:
        torch.manual_seed(4)
        training = sample_native_dropout(
            mode=mode,
            calibrated_log_variance=uncertainty,
            training=True,
        )
        evaluation = sample_native_dropout(
            mode=mode,
            calibrated_log_variance=uncertainty,
            training=False,
        )
        assert set(training) == {"predicted", "native"}
        assert all(
            torch.equal(evaluation[source][expert], torch.ones(32))
            for source in evaluation
            for expert in EXPERT_ORDER
        )
        if mode == "ND1_FIXED":
            values = list(training["native"].values())
            assert all(torch.equal(values[0], value) for value in values[1:])


@pytest.mark.parametrize("variant", REFINER_VARIANTS)
def test_refiner_variants_emit_gated_residuals(variant: str) -> None:
    predicted, native, uncertainty, _ = _evidence()
    model = NativeConditionedTokenRefiner(
        variant=variant,
        bank_dimensions=DIMS,
        token_counts=COUNTS,
        uncertainty_widths=UNCERTAINTY,
    )
    output = model(
        predicted_banks=predicted,
        calibrated_log_variance=uncertainty,
        native_banks=None if variant == "TR0_NONE" else native,
    )
    assert set(output["refined_banks"]) == set(EXPERT_ORDER)
    if variant == "TR0_NONE":
        assert output["identity_control"]
        assert all(
            torch.equal(output["refined_banks"][expert], predicted[expert])
            for expert in EXPERT_ORDER
        )
    else:
        assert all(
            output["gates"][expert].shape == predicted[expert].shape
            and bool(
                (
                    (output["gates"][expert] >= 0)
                    & (output["gates"][expert] <= 1)
                ).all()
            )
            for expert in EXPERT_ORDER
        )


@pytest.mark.parametrize("variant", ADAPTER_VARIANTS)
def test_adapter_has_exact_identity_initialization(variant: str) -> None:
    predicted, native, uncertainty, _ = _evidence()
    model = HLTResidualAdapter(
        variant=variant,
        native_dropout_mode="ND0_NONE",
        bank_dimensions=DIMS,
        token_counts=COUNTS,
        uncertainty_widths=UNCERTAINTY,
    )
    frozen = torch.randn(4, 10)
    output = model(
        frozen_offline_logits=frozen,
        predicted_banks=predicted,
        calibrated_log_variance=uncertainty,
        native_banks=native,
    )
    assert model.gamma.item() == 0.0
    assert torch.equal(output["combined_logits"], frozen)
    assert output["residual_path_logits"].shape == (4, 10)


@pytest.mark.parametrize(
    "variant", UNRESTRICTED_EVIDENCE_VARIANTS
)
def test_unrestricted_variants_and_matched_control(variant: str) -> None:
    predicted, native, uncertainty, logits = _evidence()
    model = UnrestrictedHLTFusion(
        evidence_variant=variant,
        native_dropout_mode="ND0_NONE",
        bank_dimensions=DIMS,
        token_counts=COUNTS,
        uncertainty_widths=UNCERTAINTY,
    )
    output = model(
        token_banks=predicted,
        calibrated_log_variance=uncertainty,
        native_banks=native,
        native_expert_logits=logits,
        predicted_expert_logits=logits,
    )
    assert output["logits"].shape == (4, 10)
    if variant == "F_TOKEN_ONLY_MATCHED":
        selection = output["matched_width_selection"]
        assert selection == select_matched_token_mlp_width(token_count=28)
        assert selection["selected_hidden_width"] in {
            64,
            128,
            192,
            256,
            320,
            384,
        }
    if variant == "F_TOKEN_PLUS_EXPERT_LOGITS":
        with pytest.raises(ValueError, match="deployable logits"):
            model(
                token_banks=predicted,
                calibrated_log_variance=uncertainty,
                native_banks=native,
            )


def test_eval_native_drop_is_exact_and_covers_logit_tokens() -> None:
    predicted, native, uncertainty, logits = _evidence(batch=4)
    model = UnrestrictedHLTFusion(
        evidence_variant="F_TOKEN_PLUS_EXPERT_LOGITS",
        native_dropout_mode="ND2_CONFIDENCE",
        bank_dimensions=DIMS,
        token_counts=COUNTS,
        uncertainty_widths=UNCERTAINTY,
    ).eval()
    torch.manual_seed(1)
    output = model(
        token_banks=predicted,
        calibrated_log_variance=uncertainty,
        native_banks=native,
        native_expert_logits=logits,
        predicted_expert_logits=logits,
        bypass_control="NATIVE_BRANCH_DROPPED_AT_EVALUATION",
    )
    assert all(
        torch.equal(value, torch.zeros(4))
        for value in output["availability"]["native"].values()
    )
    assert all(
        torch.equal(value, torch.ones(4))
        for value in output["availability"]["predicted"].values()
    )
    # Fourteen bank-token gates followed by fourteen logit-token gates.
    logit_gates = output["reliability_gates"][:, 28:]
    assert torch.equal(logit_gates[:, :7], torch.zeros(4, 7, 1))
    assert torch.equal(logit_gates[:, 7:], torch.ones(4, 7, 1))


def _dataset(split: str, count: int = 16) -> FinalConsumerDataset:
    rng = np.random.default_rng(1210 if split == "model_train" else 1211)
    identities = [f"{split}-{index:04d}" for index in range(count)]
    training = split == "model_train"

    def base_bank():
        return rng.normal(size=(count, 2, 64)).astype(np.float32)

    def replicas(value):
        return (
            np.stack(
                [
                    value + np.float32(index * 0.001)
                    for index in range(4)
                ]
            )
            if training
            else value
        )

    predicted = {expert: base_bank() for expert in EXPERT_ORDER}
    native = {expert: base_bank() for expert in EXPERT_ORDER}
    oracle = {expert: base_bank() for expert in EXPERT_ORDER}
    uncertainty = {
        expert: rng.normal(size=(count, 2, 1)).astype(np.float32)
        for expert in EXPERT_ORDER
    }
    heads = {expert: _Head() for expert in EXPERT_ORDER}
    fusion = _Fusion()
    with torch.no_grad():
        target_logits = {
            expert: heads[expert](torch.from_numpy(oracle[expert])).numpy()
            for expert in EXPERT_ORDER
        }
        oracle_logits = fusion(
            token_banks={
                expert: torch.from_numpy(oracle[expert])
                for expert in EXPERT_ORDER
            }
        ).numpy()
    replicas_declared = np.asarray(
        [
            replica_for(
                policy="R_MULTI",
                logical_role=split,
                epoch=0,
                canonical_identity=identity,
            )
            for identity in identities
        ]
    )
    hashes = (
        np.asarray(
            [
                [
                    _digest(f"{split}-{replica}-{identity}")
                    for identity in identities
                ]
                for replica in range(4)
            ]
        )
        if training
        else np.asarray(
            [_digest(f"{split}-0-{identity}") for identity in identities]
        )
    )
    lineage = {
        "identity_manifest": _digest("identity"),
        "HLT_view_cache": _digest("view"),
        "joint_prediction_cache": _digest("joint"),
        "native_HLT_cache": _digest("native"),
        "offline_target_cache": _digest("target"),
        "target_normalizer_set": _digest("normalizer"),
        "uncertainty_calibration": _digest("calibration"),
    }
    return FinalConsumerDataset(
        identities=identities,
        labels=np.arange(count) % 10,
        replica_ids=replicas_declared,
        degraded_view_hashes=hashes,
        split=split,
        predicted_banks={
            expert: replicas(predicted[expert])
            for expert in EXPERT_ORDER
        },
        calibrated_log_variance={
            expert: replicas(uncertainty[expert])
            for expert in EXPERT_ORDER
        },
        native_banks={
            expert: replicas(native[expert]) for expert in EXPERT_ORDER
        },
        native_expert_logits={
            expert: replicas(
                rng.normal(size=(count, 10)).astype(np.float32)
            )
            for expert in EXPERT_ORDER
        },
        predicted_expert_logits={
            expert: replicas(
                rng.normal(size=(count, 10)).astype(np.float32)
            )
            for expert in EXPERT_ORDER
        },
        oracle_banks=oracle,
        target_normalized_banks=oracle,
        target_expert_logits=target_logits,
        oracle_fusion_logits=oracle_logits,
        lineage_hashes=lineage,
    )


def test_miniature_robust_fusion_training_is_fixed_and_reusable(
    tmp_path,
) -> None:
    train = _dataset("model_train")
    stop = _dataset("val_stop")
    model = _Fusion()
    heads = {expert: _Head() for expert in EXPERT_ORDER}
    frozen_fusion = _Fusion()
    run = with_content_hash(
        {
            "contract": "retb_test_final_consumer_run_v1",
            "run_id": "RETB_OF_ROBUST_TEST",
            "consumer_kind": "OF_ROBUST",
            "model_variant": "OF_ROBUST",
            "pipeline_seed": 101,
        }
    )
    kwargs = {
        "model": model,
        "train_loader": make_final_consumer_loader(
            train, batch_size=4, seed=101, training=True
        ),
        "val_stop_loader": make_final_consumer_loader(
            stop, batch_size=4, seed=101, training=False
        ),
        "frozen_expert_heads": heads,
        "frozen_offline_fusion": frozen_fusion,
        "output_dir": tmp_path,
        "run_record": run,
        "step12_bundle_sha256": SHA_A,
        "global_determinism_sha256": SHA_B,
        "lineage_hashes": {"fixture": SHA_C},
        "source_snapshot": SOURCE,
        "config": FinalConsumerTrainingConfig(
            seed=101,
            consumer_kind="OF_ROBUST",
            model_variant="OF_ROBUST",
            maximum_epochs=2,
            microbatch_size=4,
            gradient_accumulation_steps=2,
            effective_batch_size=8,
            campaign_profile="miniature_test",
        ),
        "device": "cpu",
    }
    first = train_final_consumer(**kwargs)
    second = train_final_consumer(**kwargs)
    assert first == second
    assert first["run_record_sha256"] == run["content_hash"]
    reloaded = _Fusion()
    loaded_registration = load_selected_final_consumer_checkpoint(
        model=reloaded,
        registration_path=tmp_path / "registration.json",
        checkpoint_path=tmp_path / "best_model_val.pt",
        expected_run_record_sha256=run["content_hash"],
        expected_source=SOURCE,
    )
    assert loaded_registration == first
    assert all(
        torch.equal(reloaded.state_dict()[name], model.state_dict()[name])
        for name in model.state_dict()
    )
    evaluation = evaluate_final_consumer(
        model=model,
        consumer_kind="OF_ROBUST",
        loader=kwargs["val_stop_loader"],
        frozen_expert_heads=heads,
        frozen_offline_fusion=frozen_fusion,
        device=torch.device("cpu"),
    )
    assert evaluation["metrics"]["event_count"] == 16


class _Frontend(torch.nn.Module):
    def forward(self, *, hlt_inputs):
        return {
            "predicted_banks": hlt_inputs["predicted_banks"],
            "calibrated_log_variance": hlt_inputs[
                "calibrated_log_variance"
            ],
            "native_banks": hlt_inputs["native_banks"],
            "native_expert_logits": hlt_inputs["native_expert_logits"],
        }


class _J5FrontendStub(torch.nn.Module):
    variant = "J5_END_TO_END"

    def _live_evidence(self, shared):
        return {
            "hlt_token_banks": shared["banks"],
            "native_hlt_logits": shared["native_logits"],
        }

    def forward(self, *, evidence):
        return {
            "predicted_tokens": {
                expert: values + 1.0
                for expert, values in evidence["hlt_token_banks"].items()
            },
            "log_variance": {
                expert: values.new_zeros(
                    values.shape[0], values.shape[1], 1
                )
                for expert, values in evidence["hlt_token_banks"].items()
            },
        }


def test_joint_bridge_deployable_frontend_applies_frozen_calibration() -> None:
    banks = {
        expert: torch.zeros(2, 2, 64) for expert in EXPERT_ORDER
    }
    logits = {
        expert: torch.zeros(2, 10) for expert in EXPERT_ORDER
    }
    frontend = JointBridgeDeployableFrontend(
        joint_graph=_J5FrontendStub(),
        calibration_offsets={
            expert: [index * 0.25]
            for index, expert in enumerate(EXPERT_ORDER)
        },
    )
    output = frontend(
        hlt_inputs={"banks": banks, "native_logits": logits}
    )
    assert set(output) == {
        "predicted_banks",
        "calibrated_log_variance",
        "native_banks",
        "native_expert_logits",
    }
    for index, expert in enumerate(EXPERT_ORDER):
        assert torch.equal(
            output["predicted_banks"][expert], banks[expert] + 1.0
        )
        assert torch.equal(
            output["calibrated_log_variance"][expert],
            torch.full((2, 2, 1), index * 0.25),
        )
    with pytest.raises(ValueError, match="forbids input"):
        frontend(
            hlt_inputs={
                "banks": banks,
                "native_logits": logits,
                "labels": torch.zeros(2),
            }
        )


def test_deployable_export_rejects_offline_or_target_inputs(tmp_path) -> None:
    predicted, native, uncertainty, logits = _evidence(batch=2)
    heads = {expert: _Head() for expert in EXPERT_ORDER}
    fusion = _Fusion()
    consumer = HLTResidualAdapter(
        variant="R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS",
        native_dropout_mode="ND0_NONE",
        bank_dimensions=DIMS,
        token_counts=COUNTS,
        uncertainty_widths=UNCERTAINTY,
    )
    graph = DeployableRetbGraph(
        frontend=_Frontend(),
        final_consumer=consumer,
        consumer_kind="HF_ADAPTER",
        frozen_offline_fusion=fusion,
        frozen_expert_heads=heads,
    )
    inputs = {
        "predicted_banks": predicted,
        "calibrated_log_variance": uncertainty,
        "native_banks": native,
        "native_expert_logits": logits,
    }
    parents = {
        name: _digest(name)
        for name in (
            "campaign_spec",
            "step12_bundle",
            "HLT_frontend_checkpoint",
            "joint_predictor_or_J_checkpoint",
            "final_consumer_checkpoint",
            "frozen_offline_fusion_checkpoint",
            "frozen_offline_expert_heads",
            "HLT_input_normalizer",
            "HLT_relation_normalizer",
            "HLT_region_normalizer",
            "degradation_profile",
            "uncertainty_calibration",
        )
    }
    manifest = export_deployable_retb_graph(
        output_dir=tmp_path,
        graph=graph,
        hlt_smoke_inputs=inputs,
        parent_hashes=parents,
        source_snapshot=SOURCE,
    )
    loaded = load_deployable_retb_graph(
        tmp_path / "deployable_retb_graph.json",
        expected_source=SOURCE,
    )
    assert loaded(hlt_inputs=inputs)["logits"].shape == (2, 10)
    with pytest.raises(ValueError, match="forbids input"):
        loaded(hlt_inputs={**inputs, "offline_target_cache": SHA_A})
    with pytest.raises(ValueError, match="forbids input"):
        loaded(hlt_inputs={**inputs, "offline_particles": predicted})
    assert not manifest["target_caches_loadable"]


def test_capacity_is_per_exact_export_and_hlt_match_is_deterministic() -> None:
    modules = {
        name: torch.nn.Linear(index + 2, index + 3)
        for index, name in enumerate(CAPACITY_COMPONENTS)
    }
    flops = {
        name: {
            1: 100 + index,
            128: 128 * (100 + index),
        }
        for index, name in enumerate(CAPACITY_COMPONENTS)
    }
    capacity = build_complete_graph_capacity(
        graph_id="GRAPH_A",
        deployment_export_sha256=SHA_A,
        component_modules=modules,
        exported_graph=torch.nn.ModuleDict(modules),
        analytical_component_flops=flops,
        measured_diagnostics={"latency_ms": {"1": 1.0, "128": 2.0}},
        source_snapshot=SOURCE,
    )
    assert set(capacity["components"]) == set(CAPACITY_COMPONENTS)
    candidates = [
        {
            "configuration": [128, 4, 8, 8, 2],
            "parameter_count": capacity["totals"]["parameter_count"] + delta,
            "inference_flops_batch1": (
                capacity["totals"]["analytical_inference_flops_batch1"]
                + delta
            ),
            "inference_flops_batch128": (
                capacity["totals"]["analytical_inference_flops_batch128"]
                + delta
            ),
        }
        for delta in (10, 20)
    ]
    selection = select_monolithic_capacity_controls(
        target_parameters=capacity["totals"]["parameter_count"],
        target_flops_batch1=capacity["totals"][
            "analytical_inference_flops_batch1"
        ],
        target_flops_batch128=capacity["totals"][
            "analytical_inference_flops_batch128"
        ],
        candidates=candidates,
        domain="hlt",
        target_complete_graph_sha256=capacity["content_hash"],
    )
    assert selection["contract"] == "retb_complete_graph_capacity_controls_v2"
    assert "H_MONO_PARAM" in selection and "H_MONO_FLOP" in selection


def test_materialized_final_consumer_run_is_fail_closed() -> None:
    registry = build_final_consumer_registry(
        step11_bundle_sha256=SHA_A,
        predictor_bundle_lock_sha256=SHA_B,
    )
    row = next(
        row
        for row in registry["rows"]
        if row["consumer_kind"] == "HF_UNRESTRICTED"
        and row["token_input"] == "TOKEN_REFINED_SELECTED"
    )
    names = {
        "model_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "model_train_R_MULTI_view_cache",
        "val_stop_R_MULTI_view_cache",
        "val_design_fixed_view_cache",
        "joint_prediction_checkpoint",
        "native_HLT_checkpoint_bundle",
        "offline_target_cache",
        "target_normalizer_set",
        "uncertainty_calibration",
        "HLT_input_normalizer",
        "HLT_relation_normalizer",
        "HLT_region_normalizer",
        "degradation_profile",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
        "selected_token_refiner",
    }
    run = materialize_final_consumer_run(
        registry_row=row,
        step12_bundle_sha256=SHA_C,
        parent_hashes={name: _digest(name) for name in names},
    )
    assert validate_materialized_final_consumer_run(run) == run[
        "content_hash"
    ]
    with pytest.raises(ValueError, match="semantics differ"):
        materialize_final_consumer_run(
            registry_row=row,
            step12_bundle_sha256=SHA_C,
            parent_hashes={
                name: _digest(name)
                for name in names
                if name != "selected_token_refiner"
            },
        )


def test_step12_entrypoint_files_are_reserved() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "teacher_logit_reco/relation_expert_token_bridge/step12.py",
        "teacher_logit_reco/relation_expert_token_bridge/final_consumers.py",
        "teacher_logit_reco/relation_expert_token_bridge/final_consumer_training.py",
        "teacher_logit_reco/relation_expert_token_bridge/deployment.py",
        "scripts/build_retb_step12_contracts.py",
        "scripts/materialize_retb_final_consumer_run.py",
        "scripts/register_retb_final_consumer_dataset.py",
        "scripts/register_retb_final_consumer_template.py",
        "scripts/train_retb_final_consumer.py",
        "scripts/evaluate_retb_final_consumer_reference.py",
        "scripts/evaluate_retb_final_consumer_bypass_controls.py",
        "scripts/export_retb_deployable_graph.py",
        "scripts/prepare_retb_final_consumer_seed.py",
        "scripts/execute_retb_final_consumer_row.py",
        "scripts/select_retb_token_refiner.py",
        "scripts/execute_retb_final_consumer_campaign.py",
        "scripts/execute_retb_deployable_export_row.py",
        "scripts/finalize_retb_deployable_exports.py",
        "scripts/execute_retb_deployable_export_campaign.py",
        "scripts/attest_retb_complete_graph_capacity.py",
        "scripts/select_retb_complete_graph_capacity_controls.py",
        "sbatch/run_retb_build_step12_contracts.sh",
        "sbatch/run_retb_materialize_final_consumer_run.sh",
        "sbatch/run_retb_register_final_consumer_dataset.sh",
        "sbatch/run_retb_register_final_consumer_template.sh",
        "sbatch/run_retb_train_final_consumer.sh",
        "sbatch/run_retb_evaluate_final_consumer_reference.sh",
        "sbatch/run_retb_evaluate_final_consumer_bypass.sh",
        "sbatch/run_retb_export_deployable_graph.sh",
        "sbatch/run_retb_attest_complete_capacity.sh",
        "sbatch/run_retb_select_complete_capacity_controls.sh",
    )
    assert all((root / path).is_file() for path in paths)
    bypass = (
        root / "sbatch/run_retb_evaluate_final_consumer_bypass.sh"
    ).read_text("utf-8")
    assert "--registration" in bypass and "--checkpoint" in bypass
    export = (
        root / "sbatch/run_retb_export_deployable_graph.sh"
    ).read_text("utf-8")
    assert "--consumer-checkpoint" in export
