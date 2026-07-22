from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from teacher_logit_reco.local_particle_residual_field import (
    A0_C250,
    ROBUST_CONDITIONS,
    STEP3_RUN_IDS,
    TPRED_BRANCH_RUN_IDS,
    ConsumerCampaignConfig,
    ContinuationBranch,
    ReplicaResult,
    RobustBridgeSampler,
    aggregate_paired_replicas,
    branch_lineage_artifact,
    build_consumer_replica_manifest,
    build_consumer_tensor_batch,
    build_continuation_batch_plan,
    capture_training_lineage,
    consumer_fields_for_run,
    consumer_run_specs,
    copy_reference_hlt_weights,
    fit_bridge_corruption_scale,
    publish_paired_replicas,
    record_step3_registry_measurements,
    restore_training_lineage,
    run_exact_step_training,
    run_tpred_continuation_branches,
    verify_initial_logit_identity,
    build_campaign_registry,
)


class _Reference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(4, 7)
        self.output = torch.nn.Linear(7, 3)

    def forward(self, value):
        return self.output(torch.tanh(self.input(value)))


class _Widened(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.part_model = torch.nn.Module()
        self.part_model.input = torch.nn.Linear(54, 7)
        self.part_model.output = torch.nn.Linear(7, 3)

    def forward(self, value, fields):
        return self.part_model.output(torch.tanh(self.part_model.input(torch.cat([value, fields], dim=1))))


def _fields(n=100, p=3):
    rng = np.random.default_rng(9)
    mask = np.ones((n, p), dtype=bool)
    mask[::7, -1] = False
    f0 = rng.normal(size=(n, p, 50)).astype(np.float32)
    truth = f0 + rng.normal(scale=0.3, size=f0.shape).astype(np.float32)
    f0[~mask] = 0
    truth[~mask] = 0
    return f0, truth, mask


def test_step3_recipe_inventory_and_exact_fairness_contract():
    config = ConsumerCampaignConfig(
        baseline_steps=20,
        bridge_finetune_steps=7,
        batch_size=8,
        evaluation_interval_steps=3,
    )
    manifest = build_consumer_replica_manifest(config)
    specs = consumer_run_specs()
    assert set(specs) == set(STEP3_RUN_IDS)
    assert manifest["paired_seed_ids"] == [101, 202, 303]
    assert specs["A0_C250"].unique_jet_count == 250_000
    assert specs["A0_C250_LONG"].budget_kind == "bridge_continuation"
    assert specs["A0_S500"].unique_jet_count == 500_000
    assert set(TPRED_BRANCH_RUN_IDS) == {
        "Tpred_continue",
        "T10_clean",
        "T10_robust",
        "T10_all50_clean",
    }
    assert config.continuation_evaluation_steps == (3, 6, 7)


def test_reference_copy_zeroes_only_new_columns_and_preserves_initial_logits():
    torch.manual_seed(4)
    reference = _Reference()
    widened = _Widened()
    report = copy_reference_hlt_weights(
        reference.state_dict(), widened, added_field_dim=50
    )
    assert report["new_field_input_entries_exact_zero"] is True
    target_weight = widened.part_model.input.weight.detach()
    assert torch.equal(target_weight[:, :4], reference.input.weight.detach())
    assert torch.count_nonzero(target_weight[:, 4:]).item() == 0
    values = torch.randn(11, 4)
    zeros = torch.zeros(11, 50)
    identity = verify_initial_logit_identity(reference(values), widened(values, zeros))
    assert identity["identity_verified"] is True
    bad = deepcopy(reference.state_dict())
    bad["input.weight"] = torch.randn(7, 5)
    with pytest.raises(ValueError, match="outside the declared"):
        copy_reference_hlt_weights(bad, _Widened(), added_field_dim=50)


def test_lineage_restores_model_optimizer_scheduler_and_shared_plan_exactly():
    torch.manual_seed(8)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    plan = build_continuation_batch_plan(
        seed_id=101,
        n_examples=17,
        batch_size=5,
        steps=5,
        evaluation_steps=(2, 5),
    )
    snapshot = capture_training_lineage(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_plan=plan,
        dropout_stream_seed=991,
        robust_sampler_seed=1991,
    )
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(10)
    restore_training_lineage(
        snapshot, model=model, optimizer=optimizer, scheduler=scheduler
    )
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())
    lineage = branch_lineage_artifact(snapshot)
    assert lineage["branch_run_ids"] == list(TPRED_BRANCH_RUN_IDS)
    assert len({row["batch_plan_sha256"] for row in lineage["branches"].values()}) == 1
    assert len({row["dropout_stream_seed"] for row in lineage["branches"].values()}) == 1


def test_all_tpred_branches_restore_terminal_state_select_on_stop_split_and_stay_in_ram():
    torch.manual_seed(18)
    terminal = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(5, 2)
    )
    terminal_optimizer = torch.optim.SGD(terminal.parameters(), lr=0.05, momentum=0.9)
    terminal_scheduler = torch.optim.lr_scheduler.StepLR(terminal_optimizer, step_size=1, gamma=0.9)
    plan = build_continuation_batch_plan(
        seed_id=101,
        n_examples=6,
        batch_size=2,
        steps=3,
        evaluation_steps=(1, 3),
    )
    snapshot = capture_training_lineage(
        model=terminal,
        optimizer=terminal_optimizer,
        scheduler=terminal_scheduler,
        batch_plan=plan,
        dropout_stream_seed=8801,
        robust_sampler_seed=8802,
    )
    x = torch.linspace(-1, 1, 18).reshape(6, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    branches = []
    for run_id in TPRED_BRANCH_RUN_IDS:
        model = deepcopy(terminal)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        robust = (
            RobustBridgeSampler(seed=8802, corruption_scale=np.ones(45, dtype=np.float32))
            if run_id == "T10_robust"
            else None
        )

        def resolve(indices, robust=robust):
            if robust is not None:
                robust.draw_conditions(len(indices))
            selected = torch.as_tensor(indices, dtype=torch.long)
            return x[selected], y[selected]

        def loss_fn(current, batch, step):
            del step
            return torch.nn.functional.cross_entropy(current(batch[0]), batch[1])

        def evaluate(current, step):
            del step
            logits = current(x)
            return {
                "accuracy": float((logits.argmax(dim=-1) == y).float().mean()),
                "cross_entropy": float(torch.nn.functional.cross_entropy(logits, y)),
            }

        branches.append(
            ContinuationBranch(
                run_id=run_id,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_resolver=resolve,
                loss_fn=loss_fn,
                evaluation_fn=evaluate,
                robust_sampler=robust,
            )
        )
    result = run_tpred_continuation_branches(
        snapshot,
        batch_plan=plan,
        branches=branches,
        seed_id=101,
    )
    reports = result["reports"]
    assert set(reports) == set(TPRED_BRANCH_RUN_IDS)
    assert all(report["optimizer_steps"] == plan.steps for report in reports.values())
    assert len({tuple(report["batch_hashes"]) for report in reports.values()}) == 1
    assert len({report["initial_global_rng_sha256"] for report in reports.values()}) == 1
    assert all(report["selected_checkpoint_step"] in plan.evaluation_steps for report in reports.values())
    assert result["audit"]["candidate_weights_residency"] == "allocation_ram_only"


def test_robust_sampler_exact_cycle_frequencies_and_pass_through_channels():
    f0, truth, mask = _fields()
    scale_artifact = fit_bridge_corruption_scale(
        [(f0, truth, mask)], parent_hashes={"source": "a" * 64, "r0": "b" * 64}
    )
    sampler = RobustBridgeSampler(
        seed=707,
        corruption_scale=scale_artifact["bridge_correction_population_std"],
    )
    sampled, diagnostics = sampler.sample(f0, truth, mask)
    assert diagnostics["batch_counts"] == {
        "exact_bridge_0.100": 60,
        "exact_f0": 20,
        "uniform_bridge_0.000_0.100": 15,
        "light_corruption_bridge_0.100": 5,
    }
    assert set(diagnostics["conditions"]) == set(ROBUST_CONDITIONS)
    np.testing.assert_array_equal(sampled[..., 45:], f0[..., 45:])
    assert np.count_nonzero(sampled[~mask]) == 0
    exact_f0_rows = np.asarray(diagnostics["conditions"]) == "exact_f0"
    np.testing.assert_array_equal(sampled[exact_f0_rows], f0[exact_f0_rows])
    state = sampler.state_dict()
    expected, expected_diagnostics = sampler.sample(f0[:7], truth[:7], mask[:7])
    resumed = RobustBridgeSampler(
        seed=707,
        corruption_scale=scale_artifact["bridge_correction_population_std"],
    )
    resumed.load_state_dict(state)
    actual, actual_diagnostics = resumed.sample(f0[:7], truth[:7], mask[:7])
    np.testing.assert_array_equal(actual, expected)
    assert actual_diagnostics["conditions"] == expected_diagnostics["conditions"]
    with pytest.raises(ValueError, match="requires its separate"):
        consumer_fields_for_run("T10_robust", f0, truth, mask)


def test_consumer_checkpoint_selection_uses_accuracy_pool_then_ce_then_earliest():
    model = torch.nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    plan = build_continuation_batch_plan(
        seed_id=101,
        n_examples=3,
        batch_size=1,
        steps=3,
        evaluation_steps=(1, 2, 3),
    )
    values = {
        1: {"accuracy": 0.80000, "cross_entropy": 0.20},
        2: {"accuracy": 0.80005, "cross_entropy": 0.30},
        3: {"accuracy": 0.79999, "cross_entropy": 0.10},
    }
    selected = {}
    report = run_exact_step_training(
        model=model,
        optimizer=optimizer,
        batch_plan=plan,
        batch_resolver=lambda indices: torch.as_tensor(indices, dtype=torch.float32).view(-1, 1),
        loss_fn=lambda current, batch, step: current(batch).square().mean(),
        evaluation_fn=lambda current, step: values[step],
        selection_metric="accuracy",
        selection_state=selected,
    )
    assert selected["best_primary_value"] == pytest.approx(0.80005)
    assert selected["step"] == 3
    assert selected["cross_entropy_value"] == pytest.approx(0.10)
    assert report["selected_checkpoint_step"] == 3


def test_exact_step_engine_and_weights_only_ordered_median_publication(tmp_path):
    torch.manual_seed(3)
    x = torch.randn(12, 2)
    y = 1.7 * x[:, :1] - 0.4 * x[:, 1:]
    plan = build_continuation_batch_plan(
        seed_id=202,
        n_examples=len(x),
        batch_size=4,
        steps=30,
        evaluation_steps=(10, 20, 30),
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

    def resolve(indices):
        index = torch.as_tensor(indices)
        return x[index], y[index]

    result = run_exact_step_training(
        model=model,
        optimizer=optimizer,
        batch_plan=plan,
        batch_resolver=resolve,
        loss_fn=lambda current, batch, step: torch.nn.functional.mse_loss(current(batch[0]), batch[1]),
    )
    assert result["optimizer_steps"] == 30
    assert result["evaluation_steps"] == [10, 20, 30]
    assert result["last_loss"] < result["first_loss"]

    replicas = []
    for seed, accuracy, ce in ((101, 0.71, 0.9), (202, 0.73, 0.8), (303, 0.72, 0.7)):
        replicas.append(
            ReplicaResult(
                run_id="T10_clean",
                seed_id=seed,
                metrics={
                    "model_val_select": {
                        "bridge_accuracy": accuracy,
                        "bridge_cross_entropy": ce,
                        "same_consumer_bridge_gain": 0.02,
                    }
                },
                weights_payload={
                    "model_state_dict": {"weight": torch.tensor([seed], dtype=torch.float32)},
                    "optimizer_state_dict": {"forbidden": True},
                    "generated_fields": torch.ones(2),
                },
            )
        )
    aggregate = aggregate_paired_replicas(
        replicas,
        primary_metric="model_val_select.bridge_accuracy",
        cross_entropy_metric="model_val_select.bridge_cross_entropy",
        gain_metric="model_val_select.same_consumer_bridge_gain",
    )
    assert aggregate["median_seed_id"] == 303
    assert aggregate["best_seed_id"] == 202
    publication = publish_paired_replicas(
        replicas,
        output_dir=tmp_path / "published",
        primary_metric="model_val_select.bridge_accuracy",
        cross_entropy_metric="model_val_select.bridge_cross_entropy",
        gain_metric="model_val_select.same_consumer_bridge_gain",
    )
    assert publication["persistent_artifacts"] == [
        "aggregate_metrics.json",
        "median_weights.pt",
        "publication.json",
    ]
    checkpoint = torch.load(publication["checkpoint"], map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in checkpoint
    assert "generated_fields" not in checkpoint
    assert checkpoint["seed_id"] == 303
    with pytest.raises(PermissionError, match="reservation"):
        publish_paired_replicas(
            replicas,
            output_dir=tmp_path / "overrun",
            primary_metric="model_val_select.bridge_accuracy",
            cross_entropy_metric="model_val_select.bridge_cross_entropy",
            gain_metric="model_val_select.same_consumer_bridge_gain",
            reservation_bytes=1,
        )
    assert not (tmp_path / "overrun").exists()


def test_tensor_batch_and_measured_registry_do_not_persist_dense_fields():
    n, particles = 4, 3
    tokens = np.zeros((n, particles, 14), dtype=np.float32)
    mask = np.ones((n, particles), dtype=bool)
    tokens[..., 0] = 1.0
    tokens[..., 3] = 1.0
    labels = np.arange(n, dtype=np.int64) % 2
    f0 = np.zeros((n, particles, 50), dtype=np.float32)
    truth = np.ones_like(f0)
    batch = build_consumer_tensor_batch(
        tokens=tokens,
        mask=mask,
        labels=labels,
        f0=f0,
        f_true=truth,
        run_id=A0_C250,
    )
    assert batch["oracle_fields"] is None
    assert batch["features"].shape == (n, 17, particles)
    assert batch["offline_tokens_enter_part_inputs"] is False
    assert batch["persistent_dense_fields_written"] is False

    payloads = {
        run_id: {"run_id": run_id, "model_state_dict": {"weight": torch.ones(2)}}
        for run_id in STEP3_RUN_IDS
    }
    updated, artifact = record_step3_registry_measurements(build_campaign_registry(), payloads)
    assert set(artifact["measured_state_bytes"]) == set(STEP3_RUN_IDS)
    statuses = {
        row["canonical_run_id"]: row["measurement_status"]
        for row in updated["runs"]
        if row["canonical_run_id"] in STEP3_RUN_IDS
    }
    assert set(statuses.values()) == {"MEASURED"}
