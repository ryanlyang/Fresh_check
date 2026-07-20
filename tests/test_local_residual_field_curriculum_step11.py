from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    ALPHA_SCHEDULE_PIECEWISE,
    LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldCurriculumScheduler,
    LocalResidualFieldCurriculumSchedulerConfig,
    LocalResidualFieldTaggerConfig,
    RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
    RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
    RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
    RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
    RESIDUAL_FIELD_SOURCE_ZERO,
)
from teacher_logit_reco.local_particle_residual_field.curriculum import _ConfidenceHeads
from teacher_logit_reco.local_particle_residual_field.curriculum_report import classify_result_row
from teacher_logit_reco.local_particle_residual_field.curriculum_train import (
    LocalResidualFieldCurriculumTrainConfig,
    _evaluate_final_test_deployable,
    compute_curriculum_batch_loss,
)
from teacher_logit_reco.local_particle_residual_field.oracle import (
    FrozenLocalResidualFieldOracleConsumer,
    FrozenOracleConsumerConfig,
)
import teacher_logit_reco.local_particle_residual_field.curriculum_train as curriculum_train_module
import teacher_logit_reco.local_particle_residual_field.report as report_module
import teacher_logit_reco.local_particle_residual_field.tagger as tagger_module


FIELD_NAMES = (
    "r0p02.delta_log_pt_sum",
    "r0p02.delta_pt_frac",
    "local_reliability_score",
)
FIELD_GROUPS = {"pt_density": (0, 1), "reliability": (2,)}


class FakePart(torch.nn.Module):
    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.config = {"input_dim": int(input_dim), "num_classes": int(num_classes)}
        self.proj = torch.nn.Conv1d(int(input_dim), int(num_classes), kernel_size=1)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        weights = mask.to(dtype=features.dtype)
        per_particle = self.proj(features)
        return (per_particle * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def fake_build_hlt_classifier(*, num_classes: int, model_size: str = "base", overrides=None):
    del model_size
    input_dim = int((overrides or {}).get("input_dim", len(PF_FEATURE_NAMES)))
    return FakePart(input_dim=input_dim, num_classes=int(num_classes))


def _batch() -> dict[str, torch.Tensor]:
    batch_size, particles = 2, 4
    raw_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    target_fields = torch.tensor(
        [
            [[2.0, 0.5, 0.9], [1.0, -0.5, 0.8], [91.0, 92.0, 93.0], [94.0, 95.0, 96.0]],
            [[-2.0, 0.2, 0.7], [-1.0, -0.2, 0.6], [0.5, 0.1, 0.5], [97.0, 98.0, 99.0]],
        ],
        dtype=torch.float32,
    )
    return {
        "points": torch.zeros(batch_size, 2, particles),
        "features": torch.zeros(batch_size, len(PF_FEATURE_NAMES), particles),
        "lorentz_vectors": torch.zeros(batch_size, 4, particles),
        "mask": raw_mask[:, None, :],
        "raw_mask": raw_mask,
        "target_fields": target_fields,
        "labels": torch.tensor([0, 1]),
    }


def _tagger(
    source: str,
    *,
    alpha: float = 1.0,
    field_names=FIELD_NAMES,
    field_groups=FIELD_GROUPS,
    source_field_indices=(),
    noise_std: float = 0.0,
    field_dropout: float = 0.0,
    group_dropout: float = 0.0,
) -> LocalResidualFieldAugmentedParT:
    field_dim = len(field_names)
    return LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=2,
            field_dim=field_dim,
            field_source=source,
            oracle_field_alpha=alpha,
            oracle_field_noise_std=noise_std,
            oracle_field_dropout=field_dropout,
            oracle_field_group_dropout=group_dropout,
            field_names=field_names,
            field_groups=field_groups,
            source_field_indices=source_field_indices,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + field_dim, num_classes=2),
    )


def _forward(model: LocalResidualFieldAugmentedParT, batch: dict[str, torch.Tensor]):
    return model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        raw_mask=batch["raw_mask"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )


def _write_oracle_checkpoint(path: Path) -> None:
    model = _tagger(RESIDUAL_FIELD_SOURCE_ORACLE_SCALED)
    base_dim = len(PF_FEATURE_NAMES)
    with torch.no_grad():
        model.part_model.proj.weight.zero_()
        model.part_model.proj.bias.zero_()
        model.part_model.proj.weight[0, base_dim, 0] = 1.0
        model.part_model.proj.weight[1, base_dim, 0] = -1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "config": {"train_split": "model_train"},
            "metrics": {"accuracy": 0.75},
            "selected_field_indices": list(range(len(FIELD_NAMES))),
            "selected_field_names": list(FIELD_NAMES),
        },
        path,
    )
    (path.parent / "teacher_config.json").write_text(
        json.dumps(
            {
                "contract": "local_residual_field_oracle_teacher_config_v1",
                "teacher_id": "Ofull",
                "field_source": RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
                "oracle_field_alpha": 1.0,
                "selected_field_indices": list(range(len(FIELD_NAMES))),
                "selected_field_names": list(FIELD_NAMES),
            }
        ),
        encoding="utf-8",
    )


def _oracle_consumer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    checkpoint = tmp_path / "Ofull" / "best_model_val.pt"
    _write_oracle_checkpoint(checkpoint)
    return FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(checkpoint=str(checkpoint), consumer_id="Ofull", alpha=1.0),
        device="cpu",
    )


def test_step11_scaled_oracle_endpoints_preserve_masks_and_match_blank_or_true_fields():
    batch = _batch()
    valid = batch["raw_mask"][:, :, None].expand_as(batch["target_fields"])

    scaled = _forward(_tagger(RESIDUAL_FIELD_SOURCE_ORACLE_SCALED, alpha=0.5).eval(), batch)
    alpha_zero = _forward(_tagger(RESIDUAL_FIELD_SOURCE_ORACLE_SCALED, alpha=0.0).eval(), batch)
    blank = _forward(_tagger(RESIDUAL_FIELD_SOURCE_ZERO).eval(), batch)
    alpha_one = _forward(_tagger(RESIDUAL_FIELD_SOURCE_ORACLE_SCALED, alpha=1.0).eval(), batch)

    assert torch.equal(scaled.residual_fields[~valid], torch.zeros_like(scaled.residual_fields[~valid]))
    assert torch.allclose(scaled.residual_fields[valid], 0.5 * batch["target_fields"][valid])
    assert torch.equal(alpha_zero.residual_fields, blank.residual_fields)
    expected_true = batch["target_fields"] * valid.to(dtype=batch["target_fields"].dtype)
    assert torch.equal(alpha_one.residual_fields, expected_true)


def test_step11_field_subset_teacher_selects_noncontiguous_physical_columns():
    batch = _batch()
    selected = (0, 2)
    selected_names = tuple(FIELD_NAMES[index] for index in selected)
    output = _forward(
        _tagger(
            RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
            field_names=selected_names,
            field_groups={"physical_subset": (0, 1)},
            source_field_indices=selected,
        ).eval(),
        batch,
    )

    expected = batch["target_fields"].index_select(-1, torch.tensor(selected))
    expected = expected * batch["raw_mask"][:, :, None]
    assert torch.equal(output.residual_fields, expected)
    assert output.diagnostics["oracle_field_transform"]["oracle_field_selected_indices"] == [0, 2]
    assert output.diagnostics["oracle_field_transform"]["oracle_field_selected_names"] == list(selected_names)


@pytest.mark.parametrize(
    ("source", "noise_std"),
    (
        (RESIDUAL_FIELD_SOURCE_ORACLE_NOISY, 0.2),
        (RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT, 0.0),
    ),
)
def test_step11_noisy_and_dropout_teachers_are_deterministic_under_seed(source: str, noise_std: float):
    batch = _batch()
    model = _tagger(
        source,
        noise_std=noise_std,
        field_dropout=0.35,
        group_dropout=0.4,
    ).train()

    torch.manual_seed(1207)
    first = _forward(model, batch).residual_fields
    torch.manual_seed(1207)
    repeated = _forward(model, batch).residual_fields
    torch.manual_seed(1208)
    changed = _forward(model, batch).residual_fields

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)


def test_step11_frozen_teacher_has_no_parameter_gradients_but_predicted_fields_do(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    consumer = _oracle_consumer(tmp_path, monkeypatch)
    batch = _batch()
    predicted = torch.zeros_like(batch["target_fields"], requires_grad=True)

    output = consumer(
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        raw_mask=batch["raw_mask"],
        true_fields=batch["target_fields"],
        predicted_fields=predicted,
    )
    output.teacher_logits_pred[:, 0].sum().backward()

    assert consumer.parameters_frozen() is True
    assert all(parameter.grad is None for parameter in consumer.model.parameters())
    assert predicted.grad is not None
    assert float(predicted.grad.abs().sum()) > 0.0
    assert output.teacher_logits_true.requires_grad is False


def test_step11_oracle_path_kd_decreases_on_a_toy_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    consumer = _oracle_consumer(tmp_path, monkeypatch)
    batch = _batch()
    predicted = torch.nn.Parameter(torch.zeros_like(batch["target_fields"]))
    optimizer = torch.optim.Adam([predicted], lr=0.2)
    weights = {"ce": 0.0, "student_kd": 0.0, "oracle_path": 1.0, "field": 0.0, "gate": 0.0, "reg": 0.0}

    def oracle_path_loss():
        teacher = consumer(
            points=batch["points"],
            features=batch["features"],
            lorentz_vectors=batch["lorentz_vectors"],
            mask=batch["mask"],
            raw_mask=batch["raw_mask"],
            true_fields=batch["target_fields"],
            predicted_fields=predicted,
        )
        output = SimpleNamespace(
            student_logits=torch.zeros(2, 2),
            oracle_true_logits=teacher.teacher_logits_true,
            oracle_pred_logits=teacher.teacher_logits_pred,
            pred_fields_raw=predicted,
            pred_fields_effective=predicted,
            field_gate_loss=None,
        )
        return compute_curriculum_batch_loss(
            output,
            batch,
            loss_weights=weights,
            student_kd_source="oracle_true",
            kd_temperature=2.0,
            field_huber_beta=0.1,
        )[0]

    initial = float(oracle_path_loss().detach())
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = oracle_path_loss()
        loss.backward()
        optimizer.step()
    final = float(oracle_path_loss().detach())

    assert final < initial * 0.1
    assert all(parameter.grad is None for parameter in consumer.model.parameters())


def test_step11_confidence_gates_are_finite_clipped_and_masked():
    heads = _ConfidenceHeads(
        hidden_dim=4,
        field_dim=3,
        mode="supervised_reliability",
        initial_prob=0.5,
        log_var_min=-6.0,
        log_var_max=4.0,
        field_groups={"a": (0,), "b": (1,), "c": (2,)},
    )
    with torch.no_grad():
        heads.field_gate_head[-1].weight.zero_()
        heads.field_gate_head[-1].bias.copy_(torch.tensor([float("inf"), float("-inf"), float("nan")]))
    mask = torch.tensor([[True, False]])
    output = heads(
        hidden=torch.zeros(1, 2, 4),
        base_fields=torch.zeros(1, 2, 3),
        mask=mask,
    )

    assert torch.isfinite(output.field_gate).all()
    assert torch.all((output.field_gate >= 0.0) & (output.field_gate <= 1.0))
    assert torch.equal(output.field_gate[:, 1], torch.zeros_like(output.field_gate[:, 1]))


def test_step11_curriculum_schedule_records_alpha_for_every_epoch(tmp_path: Path):
    selected_path = tmp_path / "selected_consumer.json"
    selected_path.write_text(
        json.dumps(
            {
                "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
                "selected_consumer_id": "Ofull",
                "selected_alpha_endpoint": 0.75,
                "selection_source": "D_alpha_eval_Ofull,D_alpha_eval_Orobust",
                "selection_reason": "strongest stable endpoint",
                "model_val_alpha_curve": {"0.0": {"accuracy": 0.70}, "0.75": {"accuracy": 0.74}},
                "stack_val_alpha_curve": {"0.0": {"accuracy": 0.69}, "0.75": {"accuracy": 0.73}},
            }
        ),
        encoding="utf-8",
    )
    config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
        {
            "alpha_schedule": ALPHA_SCHEDULE_PIECEWISE,
            "piecewise_alpha": ((0, 0.25), (2, 0.5), (4, "selected_endpoint")),
        },
        selected_consumer_path=selected_path,
    )
    scheduler = LocalResidualFieldCurriculumScheduler(config, total_epochs=6)
    report_path = tmp_path / "curriculum_schedule.json"
    report = scheduler.write_report(report_path)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert [row["alpha"] for row in report["epochs"]] == [0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
    assert loaded["epochs"] == report["epochs"]
    assert all(row["teacher"]["alpha"] == row["alpha"] for row in loaded["epochs"])


def test_step11_final_test_evaluation_is_hlt_only_and_temporarily_removes_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    class Dataset:
        def __len__(self):
            return 2

    def fake_load(config):
        captured["dataset_config"] = config
        return Dataset()

    sentinel_oracle = object()
    model = SimpleNamespace(oracle_consumer=sentinel_oracle)

    def fake_epoch(active_model, loader, **kwargs):
        del loader, kwargs
        assert active_model.oracle_consumer is None
        return {"n_jets": 2, "loss": 0.4, "accuracy": 0.75}

    monkeypatch.setattr(curriculum_train_module, "load_local_particle_residual_field_hlt_only_dataset", fake_load)
    monkeypatch.setattr(curriculum_train_module, "make_local_particle_residual_field_loader", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(curriculum_train_module, "_run_epoch", fake_epoch)
    config = LocalResidualFieldCurriculumTrainConfig(
        output_dir="unused",
        hlt_cache_dir="hlt",
        target_cache_dir="must_not_be_loaded",
        run_id="P0",
        num_classes=2,
        label_names=("a", "b"),
        confirm_final_test=True,
    )

    metrics = _evaluate_final_test_deployable(model, config, device=torch.device("cpu"))

    assert captured["dataset_config"].allow_final_test_targets is False
    assert metrics["runtime_inputs"] == "HLT_only"
    assert metrics["uses_true_fields"] is False
    assert metrics["uses_offline_particles"] is False
    assert metrics["uses_teacher_logits_at_runtime"] is False
    assert metrics["oracle_teacher_loaded"] is False
    assert metrics["field_target_cache_loaded"] is False
    assert metrics["deployable"] is True
    assert metrics["selection_allowed"] is False
    assert model.oracle_consumer is sentinel_oracle


def test_step11_report_separates_deployable_rows_from_oracle_diagnostics():
    deployable = classify_result_row(
        {"run_id": "P7a", "split": "final_test", "accuracy": 0.78},
        report={"deployable": True, "runtime_inputs": "HLT_only"},
        family="curriculum",
    )
    oracle = classify_result_row(
        {"run_id": "D_alpha_eval_Ofull", "split": "model_val", "accuracy": 0.81},
        report={"deployable": False, "runtime_inputs": "HLT_plus_true_fields", "uses_true_fields": True},
        family="oracle",
    )
    o0 = classify_result_row(
        {"run_id": "O0", "split": "model_val", "accuracy": 0.70},
        report={"deployable": False, "runtime_inputs": "HLT_plus_zero_residual_fields"},
        family="oracle",
    )
    rows = [deployable, oracle, o0]
    deployable_table = report_module._strict_deployable_rows(rows)
    oracle_table = [row for row in rows if row["result_category"] == "oracle_diagnostic"]

    assert [row["run_id"] for row in deployable_table] == ["P7a"]
    assert {row["run_id"] for row in oracle_table} == {"D_alpha_eval_Ofull", "O0"}
    assert deployable["selection_allowed"] is False
    assert all(row["deployable"] is False for row in oracle_table)
    assert not ({row["run_id"] for row in deployable_table} & {row["run_id"] for row in oracle_table})
