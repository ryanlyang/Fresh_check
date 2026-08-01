from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    build_campaign_spec,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.capacity import (
    OFFLINE_CAPACITY_CONTROL_ORDER,
    build_offline_capacity_control_registration,
    build_offline_capacity_execution_registry,
    build_offline_long_exposure_ledger,
    select_monolithic_capacity_controls,
    validate_offline_capacity_control_registration,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (
    MonolithicBase4ParticleTransformer,
    analytical_particle_transformer_flops,
    build_monolithic_grid,
    monolithic_config,
)
from teacher_logit_reco.relation_expert_token_bridge.complementarity import (
    COMPLEMENTARITY_CONTRACT,
    _safe_correlation,
    build_complementarity_report,
    build_subset_readout,
    build_subset_readout_registry,
    execute_subset_readout_screen,
    linear_cka,
    shapley_from_subset_accuracy,
)
from scripts.execute_retb_offline_shape_wave import (
    _alias_experts,
    _alias_uniform_fusions,
    _optional_source_matches,
    _stage_d_parent_allocations,
)
from scripts.train_retb_offline_capacity_controls import (
    LOCKED_OFFLINE_SHAPES_RELATIVE_PATH,
    _locked_offline_shapes_path,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (
    GroupedHeadRelationBias,
    LiveExpertFusion,
    RelationAuxiliaryHead,
    TokenTransformerFusion,
    build_fusion_model,
    configure_expert_trainability,
    cross_covariance_penalty,
    fusion_parameter_groups,
    masked_relation_auxiliary_loss,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (
    load_frozen_token_cache,
    publish_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (
    OfflineFusionTrainingConfig,
    evaluate_parameter_free_fusion,
    select_best_single_expert,
    train_frozen_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    CLASS_NAMES,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (
    OfflineExpertDataset,
    make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (
    CAPACITY_VAL_DESIGN_METRICS_CONTRACT,
    _collect_predictions,
    build_capacity_val_design_metrics,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (
    EXPERT_ORDER,
    TOKEN_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.selection import (
    OFFLINE_SHAPE_SELECTION_CONTRACT,
    UNIFORM_SHAPE_METRICS_CONTRACT,
    build_uniform_shape_metrics,
    select_heterogeneous_allocations,
    select_joint_expert_losses,
    select_offline_shapes,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (
    build_step4_bundle,
    publish_step4_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.step5 import (
    build_stage_c_run_registry,
    build_step5_bundle,
    execute_miniature_stage_c,
    publish_step5_bundle,
    resolve_expert_confirmation_training_run,
    resolve_stage_c_run,
    validate_stage_c_run_registry,
    validate_step5_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (
    STAGE_D_SHAPES,
)


def _banks(events: int = 20, k: int = 2, d: int = 64):
    rng = np.random.default_rng(5101)
    return {
        name: rng.normal(size=(events, k, d)).astype(np.float32)
        for name in EXPERT_ORDER
    }


def _logits(events: int = 20):
    rng = np.random.default_rng(5102)
    return {
        name: rng.normal(size=(events, 10)).astype(np.float32)
        for name in EXPERT_ORDER
    }


def _publish_cache(root: Path, split: str, *, k: int = 2, d: int = 64):
    events = 20
    return publish_frozen_token_cache(
        output_dir=root,
        split=split,
        pipeline_seed=101,
        shape_id="S2_128" if d == 128 else "TEST_S2_64",
        identities=np.asarray([f"jet-{index:04d}" for index in range(events)]),
        labels=np.arange(events) % 10,
        token_banks=_banks(events, k, d),
        expert_logits=_logits(events),
        expert_checkpoint_hashes={name: "a" * 64 for name in EXPERT_ORDER},
        expert_registration_hashes={name: "b" * 64 for name in EXPERT_ORDER},
        identity_manifest_sha256="c" * 64,
        label_manifest_sha256="d" * 64,
    )


def test_fusion_models_have_exact_order_shapes_and_gradients() -> None:
    torch.manual_seed(5201)
    token_banks = {
        name: torch.randn(3, 2, 64, requires_grad=True)
        for name in EXPERT_ORDER
    }
    expert_logits = {
        name: torch.randn(3, 10, requires_grad=True)
        for name in EXPERT_ORDER
    }
    dimensions = {name: 64 for name in EXPERT_ORDER}
    for variant in (
        "F_TRAINED_LOGIT_LINEAR",
        "F_POOLED_MLP",
        "F_TOKEN_TRANSFORMER",
    ):
        model = build_fusion_model(variant, bank_dimensions=dimensions)
        output = model(
            token_banks=token_banks,
            expert_logits=expert_logits,
        )
        assert output.shape == (3, 10)
        output.square().mean().backward(retain_graph=True)
        assert any(
            parameter.grad is not None for parameter in model.parameters()
        )
    best = build_fusion_model(
        "F_BEST_SINGLE",
        bank_dimensions=dimensions,
        best_single_expert="TRACK",
    )
    assert torch.equal(
        best(token_banks=token_banks, expert_logits=expert_logits),
        expert_logits["TRACK"],
    )
    transformer = TokenTransformerFusion(bank_dimensions=dimensions)
    assert transformer.whole_bank_dropout_probability == 0.0
    assert len(transformer.blocks) == 3
    assert transformer.blocks[0].attention.dropout == 0.0


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.particle_encoder = torch.nn.Linear(2, 2)
        self.tokenizer = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 10)


def test_finetune_scopes_grouped_heads_and_crosscov() -> None:
    experts = {name: _Expert() for name in EXPERT_ORDER}
    frozen = configure_expert_trainability(
        experts, variant="F_TOKEN_TRANSFORMER"
    )
    assert not frozen["trainable"]
    light = configure_expert_trainability(
        experts, variant="F_TOKEN_TRANSFORMER_LIGHT_FINETUNE"
    )
    assert light["trainable"]
    assert all(".tokenizer." in name for name in light["trainable"])
    full = configure_expert_trainability(
        experts, variant="F_TOKEN_TRANSFORMER_FULL_FINETUNE"
    )
    assert len(full["trainable"]) == sum(
        len(list(expert.named_parameters())) for expert in experts.values()
    )
    grouped = GroupedHeadRelationBias()
    base = torch.randn(2, 2, 4, 4)
    relation = {
        name: torch.randn(2, 1, 4, 4)
        for name in ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
    }
    output = grouped(base4_bias=base, relation_biases=relation)
    assert output.shape == (2, 8, 4, 4)
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.equal(output[:, 2], base[:, 0] + relation["PT"][:, 0])
    penalty = cross_covariance_penalty(
        {name: torch.randn(5, 2, 4) for name in EXPERT_ORDER}
    )
    assert penalty.ndim == 0 and penalty >= 0
    auxiliary = RelationAuxiliaryHead(
        token_dimension=4, summary_dimension=3
    )
    prediction = auxiliary(torch.randn(2, 3, 4))
    target = torch.randn(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    loss = masked_relation_auxiliary_loss(prediction, target, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)
    live_experts = {name: _Expert() for name in EXPERT_ORDER}
    fusion = TokenTransformerFusion(
        bank_dimensions={name: 64 for name in EXPERT_ORDER}
    )
    live = LiveExpertFusion(
        experts=live_experts,
        fusion=fusion,
        variant="F_TOKEN_TRANSFORMER_LIGHT_FINETUNE",
    )
    groups = fusion_parameter_groups(live, fusion_learning_rate=5.0e-4)
    assert [group["lr"] for group in groups] == [5.0e-4, 5.0e-5]
    assert all(
        ".tokenizer." in name for name in live.trainability["trainable"]
    )


def test_frozen_cache_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    manifest = _publish_cache(tmp_path, "model_train")
    path = tmp_path / "model_train_frozen_tokens.json"
    loaded, arrays = load_frozen_token_cache(path)
    assert loaded["content_hash"] == manifest["content_hash"]
    assert list(arrays["token_banks"]) == list(EXPERT_ORDER)
    assert arrays["labels"].shape == (20,)
    with (tmp_path / "model_train_frozen_tokens.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="bytes differ"):
        load_frozen_token_cache(path)


def test_fixed_budget_frozen_fusion_training(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    _publish_cache(train_root, "model_train")
    _publish_cache(val_root, "val_stop")
    model = build_fusion_model(
        "F_POOLED_MLP", bank_dimensions={name: 64 for name in EXPERT_ORDER}
    )
    config = OfflineFusionTrainingConfig(
        seed=101,
        variant="F_POOLED_MLP",
        maximum_epochs=2,
        batch_size=20,
        campaign_profile="miniature_test",
    )
    registration = train_frozen_fusion(
        model=model,
        model_train_manifest=train_root / "model_train_frozen_tokens.json",
        val_stop_manifest=val_root / "val_stop_frozen_tokens.json",
        output_dir=tmp_path / "run",
        run_id="retb-test-fusion",
        run_registry_sha256="e" * 64,
        global_determinism_sha256="f" * 64,
        fusion_architecture_sha256="1" * 64,
        config=config,
        device="cpu",
    )
    assert registration["epochs_completed"] == 2
    assert registration["fixed_epoch_budget_completed"] is True
    assert registration["performance_based_termination"] is False
    assert registration["expert_parameters_updated"] is False
    assert registration["retained_checkpoints"] == ["best_model_val.pt"]
    reused = train_frozen_fusion(
        model=model,
        model_train_manifest=train_root / "model_train_frozen_tokens.json",
        val_stop_manifest=val_root / "val_stop_frozen_tokens.json",
        output_dir=tmp_path / "run",
        run_id="retb-test-fusion",
        run_registry_sha256="e" * 64,
        global_determinism_sha256="f" * 64,
        fusion_architecture_sha256="1" * 64,
        config=config,
        device="cpu",
    )
    assert reused == registration


def test_parameter_free_fusion_controls_are_executable(tmp_path: Path) -> None:
    val_root = tmp_path / "val"
    design_root = tmp_path / "design"
    _publish_cache(val_root, "val_stop")
    _publish_cache(design_root, "val_design")
    selection = select_best_single_expert(
        val_stop_manifest=val_root / "val_stop_frozen_tokens.json"
    )
    assert selection["selected_expert"] in EXPERT_ORDER
    assert selection["split"] == "val_stop"
    best = evaluate_parameter_free_fusion(
        cache_manifest=design_root / "val_design_frozen_tokens.json",
        output_path=tmp_path / "best.json",
        run_id="best-control",
        variant="F_BEST_SINGLE",
        best_single_selection=selection,
    )
    mean = evaluate_parameter_free_fusion(
        cache_manifest=design_root / "val_design_frozen_tokens.json",
        output_path=tmp_path / "mean.json",
        run_id="mean-control",
        variant="F_UNIFORM_LOGIT_MEAN",
    )
    assert best["parameter_updates"] == mean["parameter_updates"] == 0
    assert best["selected_expert"] == selection["selected_expert"]
    assert mean["selected_expert"] is None


def test_subset_registry_complementarity_and_exact_shapley() -> None:
    registry = build_subset_readout_registry(
        shape_id="S8_128", pipeline_seed=101
    )
    assert registry["subset_count"] == 128
    readout = build_subset_readout(
        mask=3,
        kind="SUBSET_LOGIT_LINEAR",
        bank_dimensions={name: 64 for name in EXPERT_ORDER},
    )
    assert readout(expert_logits={
        name: torch.randn(2, 10) for name in EXPERT_ORDER
    }).shape == (2, 10)
    completion = execute_subset_readout_screen(
        registry,
        executor=lambda row, kind: {"status": "completed"},
    )
    assert completion["readout_run_count"] == 255
    labels = np.arange(20) % 10
    logits = _logits()
    tokens = _banks(d=64)
    subset_metrics = {
        mask: {
            "accuracy": 0.1 + 0.01 * mask.bit_count(),
            "cross_entropy": 2.0 - 0.01 * mask.bit_count(),
        }
        for mask in range(128)
    }
    report = build_complementarity_report(
        shape_id="S2_64",
        pipeline_seed=101,
        cache_manifest_sha256="2" * 64,
        logits_by_expert=logits,
        labels=labels,
        tokens_by_expert=tokens,
        subset_metrics=subset_metrics,
    )
    assert report["contract"] == COMPLEMENTARITY_CONTRACT
    assert report["schema_version"] == 2
    assert report["correlation_policy"] == {
        "minimum_sample_count": 2,
        "insufficient_support_serialization": None,
        "constant_input_serialization": None,
        "mismatched_shapes": "error",
    }
    assert len(report["pairwise"]) == 21
    assert report["subset_coverage"] == 128
    assert all(
        abs(value - 0.01) < 1e-12
        for value in report["shapley_accuracy_contribution"].values()
    )
    assert linear_cka(tokens["PT"], tokens["PT"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="128"):
        shapley_from_subset_accuracy({mask: 0.0 for mask in range(127)})


def test_complementarity_correlation_handles_miniature_class_support() -> None:
    assert _safe_correlation(np.asarray([1.0]), np.asarray([0.0])) is None
    assert _safe_correlation(np.asarray([]), np.asarray([])) is None
    assert _safe_correlation(
        np.asarray([1.0, 1.0]), np.asarray([0.0, 1.0])
    ) is None
    with pytest.raises(ValueError, match="incompatible"):
        _safe_correlation(np.asarray([1.0]), np.asarray([0.0, 1.0]))


def test_shape_selector_compares_mapping_valued_sources_directly() -> None:
    source = {
        "commit": "a" * 40,
        "status_sha256": "b" * 64,
        "dirty": False,
    }
    assert _optional_source_matches(source.copy(), source)
    assert _optional_source_matches(None, source)
    assert not _optional_source_matches({**source, "dirty": True}, source)


def test_capacity_controls_consume_declared_stage_c_shape_lock(
    tmp_path: Path,
) -> None:
    expected = (
        tmp_path / "selection" / "stage_c" / "locked_offline_shapes.json"
    )
    assert LOCKED_OFFLINE_SHAPES_RELATIVE_PATH == Path(
        "selection/stage_c/locked_offline_shapes.json"
    )
    assert _locked_offline_shapes_path(tmp_path) == expected
    assert _locked_offline_shapes_path(tmp_path) != (
        tmp_path / "selection" / "locked_offline_shapes.json"
    )


class _CapacityIdentityFixture(torch.nn.Module):
    def forward(self, *, raw_tokens, **_unused):
        class_indices = raw_tokens[:, 0, 10].round().long()
        return torch.nn.functional.one_hot(
            class_indices, num_classes=10
        ).float() * 5.0


def test_capacity_prediction_uses_production_collator_identity_key() -> None:
    events = 10
    tokens = np.zeros((events, 2, 14), dtype=np.float32)
    mask = np.zeros((events, 2), dtype=bool)
    mask[:, 0] = True
    tokens[:, 0, 0] = 1.0
    tokens[:, 0, 3] = 1.0
    tokens[:, 0, 5] = 1.0
    tokens[:, 0, 10] = np.arange(events, dtype=np.float32)
    identities = [f"capacity-event-{index}" for index in range(events)]
    dataset = OfflineExpertDataset(
        tokens=tokens,
        mask=mask,
        labels=np.arange(events, dtype=np.int64),
        identities=identities,
    )
    loader = make_offline_expert_loader(
        dataset,
        seed=101,
        training=False,
        batch_size=4,
    )
    first_batch = next(iter(loader))
    assert "event_identities" in first_batch
    assert "identities" not in first_batch

    prediction = _collect_predictions(
        _CapacityIdentityFixture(), loader, device=torch.device("cpu")
    )
    assert prediction["identities"] == identities
    assert prediction["labels"].tolist() == list(range(events))
    assert prediction["metrics"]["accuracy"] == pytest.approx(1.0)

    metrics = build_capacity_val_design_metrics(
        classification_metrics=prediction["metrics"],
        control_id="O_BASE",
        checkpoint_sha256="a" * 64,
    )
    assert metrics["contract"] == CAPACITY_VAL_DESIGN_METRICS_CONTRACT
    assert metrics["classification_metrics_sha256"] == prediction[
        "metrics"
    ]["content_hash"]
    assert metrics["checkpoint_sha256"] == "a" * 64
    assert metrics["accuracy"] == pytest.approx(1.0)

    composite = build_capacity_val_design_metrics(
        classification_metrics=prediction["metrics"],
        control_id="O_7X_UNBIASED_ENSEMBLE",
        checkpoint_sha256=None,
    )
    assert composite["checkpoint_sha256"] is None

    tampered = copy.deepcopy(prediction["metrics"])
    tampered["accuracy"] = 0.0
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_capacity_val_design_metrics(
            classification_metrics=tampered,
            control_id="O_BASE",
            checkpoint_sha256="a" * 64,
        )


def _shape_rows(negative: bool = True):
    rows = []
    for shape_index, shape_id in enumerate(TOKEN_SHAPES):
        for seed in (101, 202, 303):
            accuracy = 0.30 + shape_index * 0.001
            rows.append(
                {
                    "shape_id": shape_id,
                    "pipeline_seed": seed,
                    "split": "val_design",
                    "fusion_variant": "F_TOKEN_TRANSFORMER",
                    "accuracy": accuracy,
                    "cross_entropy": 2.0 - shape_index * 0.001,
                    "per_class_efficiency": {
                        name: accuracy for name in CLASS_NAMES
                    },
                    "fusion_checkpoint_sha256": "3" * 64,
                    "fusion_registration_sha256": "4" * 64,
                    "frozen_cache_sha256": "5" * 64,
                    "metrics_artifact_sha256": "6" * 64,
                    "label_manifest_sha256": "7" * 64,
                }
            )
    return rows


def test_complete_shape_selection_emits_on_fully_negative_fixture() -> None:
    rows = _shape_rows()
    metrics = build_uniform_shape_metrics(
        rows=rows,
        stage_c_run_registry_sha256="8" * 64,
        val_design_label_manifest_sha256="7" * 64,
    )
    selection = select_offline_shapes(
        metrics, baseline_mean_accuracy=0.99
    )
    assert metrics["contract"] == UNIFORM_SHAPE_METRICS_CONTRACT
    assert metrics["schema_version"] == 2
    assert metrics["class_order"] == list(CLASS_NAMES)
    assert list(metrics["rows"][0]["per_class_efficiency"]) == list(
        CLASS_NAMES
    )
    assert selection["contract"] == OFFLINE_SHAPE_SELECTION_CONTRACT
    assert selection["schema_version"] == 2
    assert selection["all_multi_expert_models_worse_than_baseline"] is True
    assert selection["SHAPE_HIGH"]["shape_id"] in TOKEN_SHAPES
    assert selection["SHAPE_COMPACT"]["shape_id"] in TOKEN_SHAPES
    assert selection["selection_emitted_despite_scientific_result"] is True
    assert len(selection["carried_shapes_duplicate_free"]) == len(
        set(selection["carried_shapes_duplicate_free"])
    )
    with pytest.raises(ValueError, match="21"):
        build_uniform_shape_metrics(
            rows=rows[:-1],
            stage_c_run_registry_sha256="8" * 64,
            val_design_label_manifest_sha256="7" * 64,
        )
    incomplete = copy.deepcopy(metrics)
    incomplete["rows"] = incomplete["rows"][:-1]
    incomplete.pop("content_hash")
    incomplete = with_content_hash(incomplete)
    with pytest.raises(ValueError, match="21|incomplete"):
        select_offline_shapes(incomplete)


def test_uniform_shape_metrics_require_authoritative_class_mapping() -> None:
    rows = _shape_rows()
    rows[0]["per_class_efficiency"] = {
        name: rows[0]["accuracy"] for name in reversed(CLASS_NAMES)
    }
    normalized = build_uniform_shape_metrics(
        rows=rows,
        stage_c_run_registry_sha256="8" * 64,
        val_design_label_manifest_sha256="7" * 64,
    )
    assert list(normalized["rows"][0]["per_class_efficiency"]) == list(
        CLASS_NAMES
    )

    non_mapping = _shape_rows()
    non_mapping[0]["per_class_efficiency"] = [0.1] * 10
    with pytest.raises(ValueError, match="class-name mapping"):
        build_uniform_shape_metrics(
            rows=non_mapping,
            stage_c_run_registry_sha256="8" * 64,
            val_design_label_manifest_sha256="7" * 64,
        )

    incomplete = _shape_rows()
    incomplete[0]["per_class_efficiency"].pop(CLASS_NAMES[-1])
    with pytest.raises(ValueError, match="canonical class order"):
        build_uniform_shape_metrics(
            rows=incomplete,
            stage_c_run_registry_sha256="8" * 64,
            val_design_label_manifest_sha256="7" * 64,
        )


def test_joint_loss_beam_uses_fresh_tuple_readouts() -> None:
    eligible = {
        name: ["ELOSS_CE", "ELOSS_BASE_LOW"] for name in EXPERT_ORDER
    }
    individual = {
        name: {
            variant: {
                "accuracy": 0.5 + (variant == "ELOSS_BASE_LOW") * 0.001,
                "cross_entropy": 1.0,
            }
            for variant in eligible[name]
        }
        for name in EXPERT_ORDER
    }
    calls = []
    def pooled(values, seed):
        calls.append(("pooled", values, seed))
        count = values.count("ELOSS_BASE_LOW")
        return {
            "accuracy": 0.4 + count * 0.001,
            "cross_entropy": 2.0 - count * 0.001,
            "measured_flops": 10,
            "parameter_count": 10,
            "readout_sha256": f"{count + 1:x}" * 64,
        }
    def transformer(values, seed):
        calls.append(("transformer", values, seed))
        count = values.count("ELOSS_BASE_LOW")
        return {
            "accuracy": 0.5 + count * 0.001,
            "cross_entropy": 1.0 - count * 0.001,
            "fusion_sha256": f"{count + 1:x}" * 64,
        }
    selection = select_joint_expert_losses(
        eligible_variants=eligible,
        individual_metrics=individual,
        pooled_scorer=pooled,
        transformer_scorer=transformer,
        shape_id="S8_128",
    )
    assert len(selection["selected_tuple"]) == 7
    assert selection["beam_width"] == 16
    assert selection["all_CE_control_present"] is True
    assert all(call[2] == 41701 for call in calls)


def test_heterogeneous_greedy_and_beam_respect_56_slots() -> None:
    scorer_seeds = []
    def pooled(allocation, seed):
        scorer_seeds.append(seed)
        total = sum(allocation.values())
        weighted = sum(
            (index + 1) * allocation[name]
            for index, name in enumerate(EXPERT_ORDER)
        )
        return {
            "accuracy": 0.2 + weighted * 1e-5,
            "cross_entropy": 2.0 - weighted * 1e-6,
            "readout_sha256": "a" * 64,
        }
    def transformer(allocation, seed):
        total = sum(allocation.values())
        return {
            "accuracy": 0.5 + total * 1e-5,
            "cross_entropy": 1.0 - total * 1e-6,
            "fusion_sha256": "b" * 64,
        }
    selection = select_heterogeneous_allocations(
        greedy_scorer=pooled,
        beam_pooled_scorer=pooled,
        beam_transformer_scorer=transformer,
    )
    assert sum(selection["HET_PHYSICS"].values()) == 56
    assert selection["HET_SELECTED"]["total_slots"] <= 56
    assert selection["HET_BEAM"]["total_slots"] <= 56
    assert selection["beam_width"] == 32
    assert set(selection["greedy_pipeline_seeds"]) == {101, 202, 303}
    assert {101, 202, 303, 41702}.issubset(set(scorer_seeds))


def test_capacity_selectors_and_label_exposure_ledger() -> None:
    candidates = [
        {
            "configuration": [128, 4, 8, 8, 2],
            "parameter_count": 1000,
            "inference_flops_batch1": 2000,
            "inference_flops_batch128": 200000,
        },
        {
            "configuration": [256, 2, 8, 6, 1],
            "parameter_count": 1100,
            "inference_flops_batch1": 1900,
            "inference_flops_batch128": 190000,
        },
    ]
    selection = select_monolithic_capacity_controls(
        target_parameters=1050,
        target_flops_batch1=1950,
        target_flops_batch128=195000,
        candidates=candidates,
    )
    assert selection["O_MONO_PARAM"]
    assert selection["O_MONO_FLOP"]
    ledger = build_offline_long_exposure_ledger(
        component_rows=[
            {
                "component_id": "experts",
                "component_kind": "offline_expert",
                "labeled_example_presentations": 140,
                "parent_sha256": "c" * 64,
            },
            {
                "component_id": "fusion",
                "component_kind": "primary_frozen_fusion",
                "labeled_example_presentations": 20,
                "parent_sha256": "d" * 64,
            },
        ],
        obase_effective_batch_size=128,
    )
    assert ledger["optimizer_update_budget"] == 2
    assert ledger["early_stopping"] is False


class _CapacityWeaverBlock(torch.nn.Module):
    def __init__(self, width: int, ffn_ratio: int) -> None:
        super().__init__()
        self.ffn_dim = int(width) * int(ffn_ratio)
        self.fc1 = torch.nn.Linear(width, self.ffn_dim)
        self.post_fc_norm = torch.nn.LayerNorm(self.ffn_dim)
        self.fc2 = torch.nn.Linear(self.ffn_dim, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.fc2(self.post_fc_norm(torch.nn.functional.gelu(self.fc1(inputs))))
        return inputs + update


class _CapacityWeaverTransformer(torch.nn.Module):
    def __init__(self, **configuration) -> None:
        super().__init__()
        width = int(configuration["embed_dims"][-1])
        particle_ratio = int(configuration["block_params"]["ffn_ratio"])
        class_ratio = int(configuration["cls_block_params"]["ffn_ratio"])
        self.embed = torch.nn.Linear(int(configuration["input_dim"]), width)
        self.blocks = torch.nn.ModuleList(
            _CapacityWeaverBlock(width, particle_ratio)
            for _ in range(int(configuration["num_layers"]))
        )
        self.cls_blocks = torch.nn.ModuleList(
            _CapacityWeaverBlock(width, class_ratio)
            for _ in range(int(configuration["num_cls_layers"]))
        )
        self.norm = torch.nn.LayerNorm(width)
        self.fc = torch.nn.Linear(width, int(configuration["num_classes"]))

    def forward(self, features, v=None, mask=None):
        del v
        encoded = self.embed(features.transpose(1, 2))
        for block in self.blocks:
            encoded = block(encoded)
        valid = mask.transpose(1, 2).to(encoded)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        pooled = pooled.unsqueeze(1)
        for block in self.cls_blocks:
            pooled = block(pooled)
        return self.fc(self.norm(pooled.squeeze(1)))


@pytest.mark.parametrize("expansion", (2, 4, 6))
def test_monolithic_capacity_uses_weaver_ffn_ratio_end_to_end(
    expansion: int,
) -> None:
    configuration = (32, expansion, 4, 1, 1)
    resolved = monolithic_config(configuration)
    assert resolved["block_params"]["ffn_ratio"] == expansion
    assert resolved["cls_block_params"]["ffn_ratio"] == expansion
    model = MonolithicBase4ParticleTransformer(
        configuration,
        weaver_module=SimpleNamespace(
            ParticleTransformer=_CapacityWeaverTransformer
        ),
    )
    features = torch.randn(2, 17, 5, requires_grad=True)
    vectors = torch.randn(2, 4, 5)
    mask = torch.ones(2, 1, 5, dtype=torch.bool)
    logits = model(
        points=features[:, -2:, :],
        features=features,
        lorentz_vectors=vectors,
        mask=mask,
    )
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert features.grad is not None
    expected_hidden = 32 * expansion
    for block in (*model.mod.blocks, *model.mod.cls_blocks):
        assert block.fc1.out_features == expected_hidden
        assert block.fc2.in_features == expected_hidden
        assert tuple(block.post_fc_norm.normalized_shape) == (expected_hidden,)


def test_offline_capacity_execution_registry_and_registration_are_complete() -> None:
    execution = build_offline_capacity_execution_registry()
    assert tuple(execution["control_order"]) == OFFLINE_CAPACITY_CONTROL_ORDER
    assert set(execution["recipes"]) == set(OFFLINE_CAPACITY_CONTROL_ORDER)
    assert len(build_monolithic_grid()) > 100
    assert (
        analytical_particle_transformer_flops(
            configuration=(128, 4, 8, 8, 2)
        )
        > 0
    )
    registration = build_offline_capacity_control_registration(
        control_id="O_7X_UNBIASED_ENSEMBLE",
        execution_registry_sha256=execution["content_hash"],
        checkpoint_hashes=[f"{value:064x}" for value in range(1, 8)],
        parameter_count=1_000,
        inference_flops_batch1=2_000,
        inference_flops_batch128=256_000,
        profile_sha256="8" * 64,
        labeled_example_presentations=28_000,
        label_exposure_ledger_sha256="9" * 64,
        training_artifact_hashes=[f"{value:064x}" for value in range(11, 18)],
        val_design_prediction_sha256="a" * 64,
        val_design_metrics_sha256="b" * 64,
        fixed_budget_completed=True,
    )
    assert (
        validate_offline_capacity_control_registration(registration)
        == registration["content_hash"]
    )
    assert registration["checkpoint_sha256"] is None
    assert registration["performance_based_termination"] is False


def test_stage_c_registry_and_miniature_completion() -> None:
    registry = build_stage_c_run_registry()
    assert validate_stage_c_run_registry(registry) == registry["content_hash"]
    assert registry["row_counts"]["expert_shape_seed_confirmation"] == 147
    assert (
        registry["row_counts"]["expert_confirmation_physical_training_rows"]
        == 147
    )
    confirmation = resolve_expert_confirmation_training_run(
        registry,
        run_id=registry["expert_confirmation_rows"][0]["run_id"],
    )
    assert confirmation["configuration"]["tokenizer_mode"] == "TOK_CANONICAL"
    assert confirmation["configuration"]["initialization"] == "INIT_SCRATCH"
    assert confirmation["confirmation_configuration"]["kind"] == (
        "PURE_OFFLINE_EXPERT_CONFIRMATION"
    )
    assert registry["row_counts"]["canonical_fusion_shape_seed_confirmation"] == 21
    assert registry["row_counts"]["uniform_seed101_control_memberships"] == 35
    run = resolve_stage_c_run(
        registry, run_id=registry["canonical_fusion_rows"][0]["run_id"]
    )
    assert run["configuration"]["fusion_variant"] == "F_TOKEN_TRANSFORMER"
    completion = execute_miniature_stage_c(
        registry,
        expert_executor=lambda row: {"status": "completed"},
        fusion_executor=lambda row: {
            "status": "completed",
            "performance_based_termination": False,
        },
    )
    assert completion["expert_147_complete"] is True
    assert completion["fusion_21_complete"] is True


def test_stage_c_publishes_every_declared_stage_d_parent_shape() -> None:
    selection = {
        "SHAPE_COMPACT": {"K": 2},
        "SHAPE_HIGH": {"K": 16},
    }
    heterogeneous = {
        "HET_SELECTED": {
            "allocation": {
                expert: index + 1
                for index, expert in enumerate(EXPERT_ORDER)
            }
        },
        "HET_BEAM": {
            "allocation": {
                expert: index + 2
                for index, expert in enumerate(EXPERT_ORDER)
            }
        },
    }
    allocations = _stage_d_parent_allocations(
        selection=selection,
        heterogeneous=heterogeneous,
    )
    assert tuple(allocations) == STAGE_D_SHAPES
    assert allocations["S1_128"] == {
        expert: 1 for expert in EXPERT_ORDER
    }
    assert allocations["SHAPE_COMPACT"] == {
        expert: 2 for expert in EXPERT_ORDER
    }
    assert allocations["SHAPE_HIGH"] == {
        expert: 16 for expert in EXPERT_ORDER
    }


def test_stage_c_materializes_s1_expert_and_fusion_parent_paths(
    tmp_path: Path,
) -> None:
    expert_rows = []
    for seed in (101, 202, 303):
        for expert in EXPERT_ORDER:
            run_id = f"offline-{expert}-{seed}"
            expert_rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "configuration": {
                        "shape_id": "S1_128",
                        "expert_id": expert,
                    },
                }
            )
            source = (
                tmp_path
                / "runs"
                / "stage_c"
                / "offline_experts"
                / run_id
                / f"seed_{seed}"
            )
            source.mkdir(parents=True)
            (source / "checkpoint_registration.json").write_bytes(
                f"registration-{expert}-{seed}".encode()
            )
            (source / "best_model_val.pt").write_bytes(
                f"checkpoint-{expert}-{seed}".encode()
            )
    fusion_rows = []
    for seed in (101, 202, 303):
        run_id = f"fusion-{seed}"
        fusion_rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "configuration": {"shape_id": "S1_128"},
            }
        )
        source = tmp_path / "runs" / "stage_c" / run_id
        source.mkdir(parents=True)
        for name in (
            "fusion_registration.json",
            "best_model_val.pt",
            "val_design_inference.json",
            "val_design_predictions.npz",
        ):
            (source / name).write_bytes(f"{name}-{seed}".encode())
    registry = {
        "expert_confirmation_rows": expert_rows,
        "canonical_fusion_rows": fusion_rows,
    }
    _alias_experts(
        root=tmp_path,
        alias="S1_128",
        allocation={expert: 1 for expert in EXPERT_ORDER},
        registry=registry,
    )
    _alias_uniform_fusions(
        root=tmp_path,
        alias="S1_128",
        source_shape="S1_128",
        registry=registry,
    )
    for seed in (101, 202, 303):
        for expert in EXPERT_ORDER:
            target = (
                tmp_path
                / "selection"
                / "offline_experts"
                / "S1_128"
                / expert
                / f"seed_{seed}"
            )
            assert (target / "checkpoint_registration.json").is_file()
            assert (target / "best_model_val.pt").is_file()
        fusion = (
            tmp_path
            / "selection"
            / "offline_fusions"
            / "S1_128"
            / f"seed_{seed}"
        )
        assert (fusion / "fusion_registration.json").is_file()
        assert (fusion / "best_model_val.pt").is_file()


def test_step5_bundle_publication_and_selection_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.build_retb_step5_contracts import main as build_main
    from scripts.select_retb_offline_shapes import main as select_main

    snapshot = source_snapshot(Path(__file__).resolve().parents[1])
    parent_names = (
        "artifact_layout",
        "final_select_label_manifest",
        "global_determinism",
        "hlt_replica_manifest",
        "raw_input_schema",
        "scale_train_manifest",
        "split_audit",
        "split_manifest",
        "storage_measurements",
        "validation_partition_manifest",
    )
    campaign = build_campaign_spec(
        campaign_id="retb-step5-test",
        campaign_profile="miniature_test",
        source_snapshot=snapshot,
        parent_artifact_hashes={
            name: f"{index:x}"[-1] * 64
            for index, name in enumerate(parent_names, start=1)
        },
        run_registry_hashes={"runs": "f" * 64},
    )
    write_immutable_json(tmp_path / "campaign_spec.json", campaign)
    step4 = build_step4_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step3_bundle_sha256="a" * 64,
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=snapshot,
    )
    publish_step4_bundle(campaign_root=tmp_path, bundle=step4)
    step5 = build_step5_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step4_bundle_sha256=step4["step4_bundle"]["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=snapshot,
    )
    digest = validate_step5_bundle(step5)
    publication = publish_step5_bundle(campaign_root=tmp_path, bundle=step5)
    assert publication["step5_bundle_sha256"] == digest
    assert build_main(["--campaign-root", str(tmp_path), "--dry-run"]) == 0
    assert "step5_bundle_sha256" in capsys.readouterr().out
    metrics = build_uniform_shape_metrics(
        rows=_shape_rows(),
        stage_c_run_registry_sha256=step5["stage_c_run_registry"][
            "content_hash"
        ],
        val_design_label_manifest_sha256="7" * 64,
    )
    metrics = bind_source(metrics, source_snapshot=snapshot)
    metrics_path = tmp_path / "metrics" / "uniform_shapes.json"
    write_immutable_json(metrics_path, metrics)
    assert (
        select_main(
            [
                "--campaign-root",
                str(tmp_path),
                "--uniform-metrics",
                str(metrics_path),
                "--baseline-mean-accuracy",
                "0.99",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"all_multi_expert_models_worse_than_baseline": true' in output
