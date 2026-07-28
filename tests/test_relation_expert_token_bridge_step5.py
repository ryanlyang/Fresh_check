from __future__ import annotations

import copy
from pathlib import Path

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
    build_offline_long_exposure_ledger,
    select_monolithic_capacity_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.complementarity import (
    build_complementarity_report,
    build_subset_readout,
    build_subset_readout_registry,
    execute_subset_readout_screen,
    linear_cka,
    shapley_from_subset_accuracy,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (
    GroupedHeadRelationBias,
    RelationAuxiliaryHead,
    TokenTransformerFusion,
    build_fusion_model,
    configure_expert_trainability,
    cross_covariance_penalty,
    masked_relation_auxiliary_loss,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (
    load_frozen_token_cache,
    publish_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (
    OfflineFusionTrainingConfig,
    train_frozen_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (
    EXPERT_ORDER,
    TOKEN_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.selection import (
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
    resolve_stage_c_run,
    validate_stage_c_run_registry,
    validate_step5_bundle,
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
    assert len(report["pairwise"]) == 21
    assert report["subset_coverage"] == 128
    assert all(
        abs(value - 0.01) < 1e-12
        for value in report["shapley_accuracy_contribution"].values()
    )
    assert linear_cka(tokens["PT"], tokens["PT"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="128"):
        shapley_from_subset_accuracy({mask: 0.0 for mask in range(127)})


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
                    "per_class_efficiency": [accuracy] * 10,
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
    def pooled(allocation, seed):
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


def test_stage_c_registry_and_miniature_completion() -> None:
    registry = build_stage_c_run_registry()
    assert validate_stage_c_run_registry(registry) == registry["content_hash"]
    assert registry["row_counts"]["expert_shape_seed_confirmation"] == 147
    assert registry["row_counts"]["canonical_fusion_shape_seed_confirmation"] == 21
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
