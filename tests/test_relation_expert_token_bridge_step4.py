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
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    CLASSIFICATION_METRICS_CONTRACT,
    evaluate_classification,
    expected_calibration_error,
    qcd_signal_rejection,
    stable_probabilities,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (
    EXPERT_LOSS_CANDIDATES,
    OfflineExpertDataset,
    OfflineExpertTrainingConfig,
    apply_attachment_trainability,
    build_attachment_pretraining_record,
    build_expert_loss_registry,
    build_teacher_logits_manifest,
    copy_obase_particle_backbone,
    make_offline_expert_loader,
    offline_expert_objective,
    preferred_expert_epoch,
    train_offline_expert,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (
    OfflineClassifierAdapter,
)
from teacher_logit_reco.relation_expert_token_bridge.step3 import (
    build_step3_bundle,
    publish_step3_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (
    STEP4_MINIATURE_COMPLETION_CONTRACT,
    aggregate_optimization_candidate_metrics,
    build_full_optimization_rows,
    build_optimization_candidate_metrics,
    build_primary_shape_screen_rows,
    build_stage_b_run_registry,
    build_step4_bundle,
    execute_miniature_stage_b,
    materialize_optimization_winner_followups,
    publish_step4_bundle,
    resolve_stage_b_run,
    select_optimization_candidate,
    validate_stage_b_run_registry,
    validate_step4_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (
    CanonicalSummaryTokenizer,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


class _TinyMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(17, 16)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(16, 16) for _ in range(8)]
        )


class _TinyParticleEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = _TinyMod()
        self.relation_scale = torch.nn.Parameter(torch.ones(()))


class _TinyTokenizer(torch.nn.Module):
    def __init__(self, token_count: int, token_dimension: int) -> None:
        super().__init__()
        self.slot_queries = torch.nn.Parameter(
            torch.randn(token_count, token_dimension) * 0.01
        )
        self.projection = torch.nn.Linear(16, token_dimension)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return (
            self.projection(pooled).unsqueeze(1)
            + self.slot_queries.unsqueeze(0)
        )


class _TinyExpert(torch.nn.Module):
    def __init__(self, token_count: int, token_dimension: int) -> None:
        super().__init__()
        self.particle_encoder = _TinyParticleEncoder()
        self.tokenizer = _TinyTokenizer(token_count, token_dimension)
        self.head = torch.nn.Linear(token_dimension, 10)

    def forward(
        self,
        *,
        features,
        vectors,
        mask,
        raw_tokens,
        region_trees=None,
        return_details=False,
    ):
        del vectors, raw_tokens, region_trees
        valid = mask[:, 0].bool()
        state = torch.tanh(
            self.particle_encoder.mod.embed(features.transpose(1, 2))
        )
        for block in self.particle_encoder.mod.blocks:
            state = torch.tanh(block(state))
        state = state.masked_fill(~valid.unsqueeze(-1), 0.0)
        pooled = state.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        tokens = self.tokenizer(pooled)
        logits = self.head(tokens.mean(dim=1))
        if return_details:
            return {"tokens": tokens, "logits": logits}
        return logits


class _OrdinaryClassifier(torch.nn.Module):
    def forward(self, *, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        pooled = (features * mask).sum(dim=-1)
        return pooled[:, :10]


def test_ordinary_weaver_adapter_exposes_registered_control_semantics() -> None:
    configuration = {
        "expert_id": "PT",
        "topology": "B_CONCAT",
        "particle_dropout": 0.0,
        "measurement_embedding": False,
        "shape_id": "S8_128",
        "token_count": 8,
        "token_dimension": 128,
        "tokenizer_mode": "TOK_WEAVER_CLASS",
    }
    model = OfflineClassifierAdapter(
        _OrdinaryClassifier(), expert_configuration=configuration
    )
    output = model(
        features=torch.randn(3, 17, 5),
        vectors=torch.randn(3, 4, 5),
        mask=torch.ones(3, 1, 5, dtype=torch.bool),
        raw_tokens=torch.randn(3, 5, 14),
        region_trees={"ignored": True},
        return_details=True,
    )
    assert output["logits"].shape == (3, 10)
    assert output["tokens"].shape == (3, 8, 128)
    assert not bool(output["tokens"].any())
    assert model.particle_encoder.expert_id == "PT"
    assert model.tokenizer_mode == "TOK_WEAVER_CLASS"


def _offline_arrays(events: int = 20, particles: int = 4):
    if events % 10:
        raise ValueError("fixture must remain class balanced")
    rng = np.random.default_rng(4107)
    tokens = np.zeros((events, particles, 14), dtype=np.float32)
    mask = np.ones((events, particles), dtype=bool)
    labels = np.arange(events, dtype=np.int64) % 10
    for event in range(events):
        pt = rng.uniform(1.0, 20.0, size=particles)
        eta = rng.uniform(-1.5, 1.5, size=particles)
        phi = rng.uniform(-np.pi, np.pi, size=particles)
        mass = rng.uniform(0.0, 0.3, size=particles)
        tokens[event, :, 0] = pt
        tokens[event, :, 1] = eta
        tokens[event, :, 2] = phi
        tokens[event, :, 3] = np.sqrt(
            (pt * np.cosh(eta)) ** 2 + mass**2
        )
        for particle in range(particles):
            category = (event + particle) % 6
            if category < 5:
                tokens[event, particle, 5 + category] = 1.0
            if category in (0, 3, 4):
                tokens[event, particle, 4] = 1.0
                tokens[event, particle, 10:14] = [0.1, 0.02, 0.2, 0.04]
    return {
        "tokens": tokens,
        "mask": mask,
        "labels": labels,
        "identities": [f"offline.root#{index}" for index in range(events)],
    }


def _loaders(*, batch_size: int = 10):
    arrays = _offline_arrays()
    dataset = OfflineExpertDataset(**arrays)
    return (
        make_offline_expert_loader(
            dataset,
            seed=101,
            training=True,
            batch_size=batch_size,
        ),
        make_offline_expert_loader(
            dataset,
            seed=101,
            training=False,
            batch_size=batch_size,
        ),
    )


def test_primary_49_rows_and_all_declared_stage_b_controls_are_exact() -> None:
    primary = build_primary_shape_screen_rows()
    assert len(primary) == 49
    assert len({row["run_id"] for row in primary}) == 49
    assert {
        row["configuration"]["expert_id"] for row in primary
    } == {"BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"}
    assert all(row["configuration"]["topology"] == "B_CONCAT" for row in primary)
    assert all(row["configuration"]["loss_id"] == "ELOSS_CE" for row in primary)
    registry = build_stage_b_run_registry()
    assert validate_stage_b_run_registry(registry) == registry["content_hash"]
    assert registry["row_counts"]["primary_shape_screen"] == 49
    assert registry["row_counts"]["full_PT_TRACK_optimization_grid"] == 36
    assert registry["row_counts"]["dual_topology_controls"] == 14
    optimization = build_full_optimization_rows()
    assert len(optimization) == 36
    primary_default = {
        (
            row["configuration"]["expert_id"],
            row["configuration"]["shape_id"],
        ): row["run_id"]
        for row in primary
    }
    for row in optimization:
        configuration = row["configuration"]
        if (
            configuration["initialization"] == "INIT_SCRATCH"
            and configuration["learning_rate"] == 1.0e-3
            and configuration["particle_dropout"] == 0.0
        ):
            assert row["run_id"] == primary_default[
                (configuration["expert_id"], "S8_128")
            ]
    resolved = resolve_stage_b_run(registry, run_id=primary[3]["run_id"])
    assert resolved["configuration"] == primary[3]["configuration"]
    assert resolved["configuration"]["performance_based_termination"] is False


def test_expert_loss_candidates_have_exact_ce_kd_semantics_and_detach() -> None:
    torch.manual_seed(4501)
    logits = torch.randn(6, 10, requires_grad=True)
    labels = torch.arange(6) % 10
    obase = torch.randn(6, 10, requires_grad=True)
    full = torch.randn(6, 10, requires_grad=True)
    strongest = torch.randn(6, 10, requires_grad=True)
    teacher = {
        "O_BASE": obase,
        "O_FULLREL": full,
        "SELECTED_STRONGEST": strongest,
    }
    for loss_id, definition in EXPERT_LOSS_CANDIDATES.items():
        logits.grad = None
        teacher_name = definition["teacher"]
        selected_teacher = (
            None
            if teacher_name is None
            else {
                "O_BASE": obase,
                "O_FULLREL": full,
            }
            if teacher_name == "MEAN_PROBABILITY_O_BASE_O_FULLREL"
            else {teacher_name: teacher[teacher_name]}
        )
        loss, components = offline_expert_objective(
            logits,
            labels,
            loss_id=loss_id,
            teacher_logits=selected_teacher,
        )
        loss.backward(retain_graph=True)
        assert logits.grad is not None
        assert torch.isfinite(loss)
        if definition["kd_weight"] == 0:
            assert components["knowledge_distillation"].item() == 0
        else:
            assert components["knowledge_distillation"].item() >= 0
    assert obase.grad is None and full.grad is None and strongest.grad is None
    with pytest.raises(ValueError, match="must not consume"):
        offline_expert_objective(
            logits,
            labels,
            loss_id="ELOSS_CE",
            teacher_logits={"O_BASE": obase},
        )
    with pytest.raises(ValueError, match="requires only teacher"):
        offline_expert_objective(
            logits,
            labels,
            loss_id="ELOSS_BASE",
            teacher_logits={"O_BASE": obase, "O_FULLREL": full},
        )
    manifest = build_teacher_logits_manifest(
        model_train_npz_sha256="1" * 64,
        val_stop_npz_sha256="2" * 64,
        teacher_checkpoint_hashes={"O_BASE": "3" * 64},
        teacher_fields=["O_BASE"],
    )
    assert manifest["teacher_fields"] == ["O_BASE"]


def test_obase_initialization_copies_only_particle_backbone() -> None:
    torch.manual_seed(4703)
    model = _TinyExpert(2, 64)
    original_query = model.tokenizer.slot_queries.detach().clone()
    source = _TinyMod().state_dict()
    source = {
        f"mod.{name}": torch.full_like(value, 0.25)
        for name, value in source.items()
    }
    report = copy_obase_particle_backbone(model, source)
    assert report["copied_tensor_count"] > 0
    assert report["relation_or_token_parameter_copied"] is False
    assert torch.equal(model.tokenizer.slot_queries, original_query)
    assert torch.equal(
        model.particle_encoder.mod.embed.weight,
        torch.full_like(model.particle_encoder.mod.embed.weight, 0.25),
    )


def test_obase_adapter_checkpoint_prefix_copies_particle_backbone() -> None:
    model = _TinyExpert(2, 64)
    source = {
        f"classifier.mod.{name}": torch.full_like(value, 0.125)
        for name, value in _TinyMod().state_dict().items()
    }
    report = copy_obase_particle_backbone(model, source)
    assert report["copied_tensor_count"] > 0
    assert torch.equal(
        model.particle_encoder.mod.embed.weight,
        torch.full_like(model.particle_encoder.mod.embed.weight, 0.125),
    )


def test_attachment_schedule_freezes_exact_backbone_regions() -> None:
    model = _TinyExpert(2, 64)
    first = apply_attachment_trainability(model, epoch=1)
    assert first["phase"] == "backbone_frozen"
    assert not model.particle_encoder.mod.embed.weight.requires_grad
    assert model.particle_encoder.relation_scale.requires_grad
    assert model.tokenizer.slot_queries.requires_grad
    middle = apply_attachment_trainability(model, epoch=6)
    assert middle["phase"] == "last_four_blocks"
    assert not model.particle_encoder.mod.blocks[3].weight.requires_grad
    assert model.particle_encoder.mod.blocks[4].weight.requires_grad
    final = apply_attachment_trainability(model, epoch=11)
    assert final["phase"] == "complete_graph"
    assert all(parameter.requires_grad for parameter in model.parameters())
    record = build_attachment_pretraining_record(
        checkpoint_sha256=SHA_A,
        epochs=40,
        label_presentations=20_000_000,
        optimizer_updates=156_280,
        walltime_seconds=100.0,
    )
    assert record["included_in_long_baseline_capacity_matching"] is True


def test_checkpoint_selector_uses_global_window_ce_and_earliest() -> None:
    rows = [
        {"epoch": 1, "val_stop": {"accuracy": 0.8, "cross_entropy": 0.7}},
        {"epoch": 2, "val_stop": {"accuracy": 0.80009, "cross_entropy": 0.6}},
        {"epoch": 3, "val_stop": {"accuracy": 0.80009, "cross_entropy": 0.6}},
    ]
    assert preferred_expert_epoch(rows)["epoch"] == 2
    rows.append(
        {"epoch": 4, "val_stop": {"accuracy": 0.8002, "cross_entropy": 0.9}}
    )
    assert preferred_expert_epoch(rows)["epoch"] == 4


def test_exact_classification_metrics_ece_and_rejection_contract() -> None:
    rng = np.random.default_rng(4909)
    logits = rng.normal(size=(40, 10))
    labels = np.arange(40) % 10
    metrics = evaluate_classification(logits, labels, split="val_stop")
    assert metrics["contract"] == CLASSIFICATION_METRICS_CONTRACT
    assert len(metrics["ece_15_bin"]["bins"]) == 15
    assert metrics["qcd_signal_rejection"]["Hbb"]["0.3"][
        "discriminant"
    ] == "p_signal/(p_signal+p_QCD)"
    probability = stable_probabilities(logits)
    rejection = qcd_signal_rejection(
        probability, labels, signal_index=1, target_efficiency=0.5
    )
    assert rejection["pass_rule"] == "score_greater_than_or_equal_to_threshold"
    ece = expected_calibration_error(probability, labels)
    assert sum(row["count"] for row in ece["bins"]) == len(labels)


def test_tokenizer_diagnostics_store_sufficient_statistics_not_attention() -> None:
    torch.manual_seed(5113)
    tokenizer = CanonicalSummaryTokenizer(
        expert_id="PT",
        token_count=4,
        token_dimension=64,
    ).eval()
    tokenizer.set_collect_attention_diagnostics(True)
    states = torch.randn(3, 6, 128)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )
    tokens = tokenizer(states, mask)
    assert tokens.shape == (3, 4, 64)
    statistics = tokenizer.attention_sufficient_statistics()
    assert len(statistics) == 2
    assert statistics[0]["event_count"] == 3
    assert "weights" not in statistics[0]
    tokenizer.set_collect_attention_diagnostics(False)


def test_fixed_budget_trainer_retains_only_selected_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(5303)
    train_loader, val_loader = _loaders()
    run = build_primary_shape_screen_rows()[0]
    configuration = run["configuration"]
    model = _TinyExpert(
        configuration["token_count"], configuration["token_dimension"]
    )
    config = OfflineExpertTrainingConfig(
        seed=101,
        maximum_epochs=2,
        microbatch_size=10,
        gradient_accumulation_steps=2,
        effective_batch_size=20,
        campaign_profile="miniature_test",
    )
    registration = train_offline_expert(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_loader,
        output_dir=tmp_path,
        run_record=run,
        run_registry_sha256=SHA_A,
        step3_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        expert_loss_registry=build_expert_loss_registry(),
        lineage_hashes={
            "campaign_spec": "d" * 64,
            "model_train_inputs": "e" * 64,
            "val_stop_inputs": "f" * 64,
        },
        config=config,
        device="cpu",
    )
    assert registration["epochs_completed"] == 2
    assert registration["fixed_epoch_budget_completed"] is True
    assert registration["stopped_early"] is False
    assert registration["performance_based_termination"] is False
    assert registration["retained_checkpoints"] == ["best_model_val.pt"]
    assert (tmp_path / "best_model_val.pt").is_file()
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / ".checkpoint_frontier").exists()
    assert (tmp_path / "val_stop_predictions.npz").is_file()
    assert registration["diagnostics_sha256"]
    repeated = train_offline_expert(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_loader,
        output_dir=tmp_path,
        run_record=run,
        run_registry_sha256=SHA_A,
        step3_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        expert_loss_registry=build_expert_loss_registry(),
        lineage_hashes={
            "campaign_spec": "d" * 64,
            "model_train_inputs": "e" * 64,
            "val_stop_inputs": "f" * 64,
        },
        config=config,
        device="cpu",
    )
    assert repeated["content_hash"] == registration["content_hash"]
    with (tmp_path / "val_stop_predictions.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="prediction bytes differ"):
        train_offline_expert(
            model=model,
            train_loader=train_loader,
            val_stop_loader=val_loader,
            output_dir=tmp_path,
            run_record=run,
            run_registry_sha256=SHA_A,
            step3_bundle_sha256=SHA_B,
            global_determinism_sha256=SHA_C,
            expert_loss_registry=build_expert_loss_registry(),
            lineage_hashes={
                "campaign_spec": "d" * 64,
                "model_train_inputs": "e" * 64,
                "val_stop_inputs": "f" * 64,
            },
            config=config,
            device="cpu",
        )


def test_miniature_primary_49_and_optimization_subset_all_execute() -> None:
    registry = build_stage_b_run_registry()
    arrays = _offline_arrays()
    dataset = OfflineExpertDataset(**arrays)
    batch = next(
        iter(
            make_offline_expert_loader(
                dataset,
                seed=101,
                training=False,
                batch_size=20,
            )
        )
    )

    def execute(row):
        configuration = row["configuration"]
        model = _TinyExpert(
            configuration["token_count"], configuration["token_dimension"]
        )
        if configuration["initialization"] == "INIT_ATTACH_AFTER_PRETRAIN":
            apply_attachment_trainability(model, epoch=1)
        logits = model(
            features=batch["features"],
            vectors=batch["vectors"],
            mask=batch["mask"],
            raw_tokens=batch["raw_tokens"],
        )
        loss, _ = offline_expert_objective(
            logits, batch["labels"], loss_id="ELOSS_CE"
        )
        loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        return {
            "status": "completed",
            "epochs_completed": 1,
            "performance_based_termination": False,
        }

    completion = execute_miniature_stage_b(
        registry,
        executor=execute,
        selected_optimization_configuration={
            "initialization": "INIT_OBASE_PARTICLE",
            "learning_rate": 5.0e-4,
            "particle_dropout": 0.1,
        },
    )
    assert completion["contract"] == STEP4_MINIATURE_COMPLETION_CONTRACT
    assert completion["primary_49_complete"] is True
    assert completion["declared_optimization_subset_complete"] is True
    assert completion["section_counts"] == {
        "primary_shape_screen": 49,
        "full_optimization_grid": 36,
        "fixed_followup_references": 4,
        "winner_followups": 2,
    }


def test_optimization_selector_always_emits_even_all_negative() -> None:
    rows = []
    for index, row in enumerate(build_full_optimization_rows()[:4]):
        rows.append(
            {
                "run_id": row["run_id"],
                "configuration": row["configuration"],
                "split": "val_design",
                "accuracy": 0.4 - index * 0.01,
                "cross_entropy": 2.0 + index,
                "measured_flops": 100 + index,
                "parameter_count": 1000 + index,
            }
        )
    selection = select_optimization_candidate(
        rows, baseline_accuracy=0.9
    )
    assert selection["selected_run_id"] == rows[0]["run_id"]
    assert selection["gain_positive"] is False
    assert selection["all_candidates_worse_than_baseline"] is True
    followups = materialize_optimization_winner_followups(
        selection["selected_configuration"]
    )
    assert [row["configuration"]["expert_id"] for row in followups] == [
        "BASE4",
        "REGION",
    ]
    with pytest.raises(ValueError, match="val_design"):
        select_optimization_candidate([{**rows[0], "split": "val_stop"}])


def test_optimization_selector_uses_complete_paired_pt_track_grid() -> None:
    registry = build_stage_b_run_registry()
    label_hash = "7" * 64
    raw = []
    for index, row in enumerate(registry["full_optimization_grid"]):
        raw.append(
            {
                "run_id": row["run_id"],
                "configuration": row["configuration"],
                "split": "val_design",
                "accuracy": 0.5 + index / 10_000,
                "cross_entropy": 1.5 - index / 10_000,
                "measured_flops": 1000 + index,
                "parameter_count": 10_000 + index,
                "checkpoint_sha256": "8" * 64,
                "prediction_shard_sha256": "9" * 64,
                "label_manifest_sha256": label_hash,
                "metrics_artifact_sha256": "a" * 64,
            }
        )
    manifest = build_optimization_candidate_metrics(
        run_registry=registry,
        rows=raw,
        val_design_label_manifest_sha256=label_hash,
    )
    candidates = aggregate_optimization_candidate_metrics(
        manifest,
        run_registry=registry,
    )
    assert len(candidates) == 18
    assert all(len(row["contributing_run_ids"]) == 2 for row in candidates)
    selection = select_optimization_candidate(candidates)
    assert selection["candidate_metrics_sha256"] == manifest["content_hash"]
    assert len(selection["selected_contributing_run_ids"]) == 2
    with pytest.raises(ValueError, match="complete grid"):
        build_optimization_candidate_metrics(
            run_registry=registry,
            rows=raw[:-1],
            val_design_label_manifest_sha256=label_hash,
        )


def test_step4_bundle_publication_and_cli_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.build_retb_step4_contracts import main
    from scripts.select_retb_expert_optimization import main as select_main
    from scripts.train_retb_offline_expert import main as train_main

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
        campaign_id="retb-step4-test",
        campaign_profile="miniature_test",
        source_snapshot=snapshot,
        parent_artifact_hashes={
            name: f"{index:x}"[-1] * 64
            for index, name in enumerate(parent_names, start=1)
        },
        run_registry_hashes={"runs": "f" * 64},
    )
    write_immutable_json(tmp_path / "campaign_spec.json", campaign)
    step3 = build_step3_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )
    publish_step3_bundle(campaign_root=tmp_path, bundle=step3)
    step4 = build_step4_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step3_bundle_sha256=step3["step3_bundle"]["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=snapshot,
    )
    digest = validate_step4_bundle(step4)
    publication = publish_step4_bundle(campaign_root=tmp_path, bundle=step4)
    assert publication["step4_bundle_sha256"] == digest
    assert (tmp_path / "registry" / "retb_stage_b_runs.json").is_file()
    assert main(["--campaign-root", str(tmp_path), "--dry-run"]) == 0
    assert "step4_bundle_sha256" in capsys.readouterr().out
    assert (
        train_main(
            [
                "--campaign-root",
                str(tmp_path),
                "--run-id",
                step4["stage_b_run_registry"]["primary_shape_screen"][0][
                    "run_id"
                ],
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"performance_based_termination": false' in capsys.readouterr().out
    registry = step4["stage_b_run_registry"]
    metric_rows = [
        {
            "run_id": row["run_id"],
            "configuration": row["configuration"],
            "split": "val_design",
            "accuracy": 0.4,
            "cross_entropy": 2.0,
            "measured_flops": 100.0,
            "parameter_count": 1000,
            "checkpoint_sha256": "1" * 64,
            "prediction_shard_sha256": "2" * 64,
            "label_manifest_sha256": "3" * 64,
            "metrics_artifact_sha256": "4" * 64,
        }
        for row in registry["full_optimization_grid"]
    ]
    metrics = build_optimization_candidate_metrics(
        run_registry=registry,
        rows=metric_rows,
        val_design_label_manifest_sha256="3" * 64,
    )
    metrics = bind_source(metrics, source_snapshot=snapshot)
    metrics_path = tmp_path / "metrics" / "optimization.json"
    write_immutable_json(metrics_path, metrics)
    assert (
        select_main(
            [
                "--campaign-root",
                str(tmp_path),
                "--candidate-metrics",
                str(metrics_path),
                "--baseline-accuracy",
                "0.9",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"all_candidates_worse_than_baseline": true' in output
    shell = (
        Path(__file__).resolve().parents[1]
        / "sbatch"
        / "run_retb_train_offline_expert.sh"
    ).read_text(encoding="utf-8")
    assert "RETB_RUN_ID" in shell
    assert "EARLY_STOP" not in shell
    assert "--dry-run" in shell
