from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FIELD_GATE_MODE_NONE,
    LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
    LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
    PILOT_LOSS_WEIGHTS,
    RESIDUAL_FIELD_SOURCE_ZERO,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldCurriculumJointConfig,
    LocalResidualFieldCurriculumJointModel,
    LocalResidualFieldCurriculumTrainConfig,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldTaggerConfig,
    compute_curriculum_batch_loss,
    resolve_curriculum_run,
    train_local_residual_field_curriculum,
)
from teacher_logit_reco.local_particle_residual_field import curriculum_train as curriculum_train_module


ROOT = Path(__file__).resolve().parents[1]
FIELD_NAMES = ("field.a", "field.b")
FIELD_GROUPS = {"pair": [0, 1]}


def _selected_consumer(path: Path, *, consumer: str = "Ofull", endpoint: float = 0.75) -> Path:
    payload = {
        "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
        "selected_consumer_id": consumer,
        "selected_alpha_endpoint": endpoint,
        "selection_source": "stage1a/selector_report.json",
        "selection_reason": "smooth useful response",
        "model_val_alpha_curve": {"0.0": {"accuracy": 0.5}, str(endpoint): {"accuracy": 0.6}},
        "stack_val_alpha_curve": {"0.0": {"accuracy": 0.5}, str(endpoint): {"accuracy": 0.6}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(tmp_path: Path, run_id: str, **kwargs) -> LocalResidualFieldCurriculumTrainConfig:
    payload = {
        "output_dir": str(tmp_path / run_id),
        "hlt_cache_dir": str(tmp_path / "hlt"),
        "target_cache_dir": str(tmp_path / "target"),
        "run_id": run_id,
        "epochs": 7,
        "num_classes": 3,
        "label_names": ("a", "b", "c"),
    }
    payload.update(kwargs)
    return LocalResidualFieldCurriculumTrainConfig(**payload)


def test_step7_stage1b_requires_selector_and_rejects_guessed_consumer(tmp_path: Path):
    with pytest.raises(ValueError, match="requires selected_consumer.json"):
        resolve_curriculum_run(_config(tmp_path, "P2"))

    selected = _selected_consumer(tmp_path / "selected_consumer.json")
    with pytest.raises(ValueError, match="differs from selected P7a consumer"):
        resolve_curriculum_run(
            _config(
                tmp_path,
                "P2",
                selected_consumer_json=str(selected),
                consumer_id="Orobust_light",
            )
        )


def test_step7_q0_and_q3_are_locked_to_selected_p7a_contract(tmp_path: Path):
    selected = _selected_consumer(tmp_path / "selected_consumer.json", endpoint=0.75)
    q0 = resolve_curriculum_run(
        _config(
            tmp_path,
            "Q0",
            selected_consumer_json=str(selected),
            student_warm_start_checkpoint="a0.pt",
        )
    )
    expected_q0 = dict(PILOT_LOSS_WEIGHTS["P7a"])
    expected_q0["oracle_path"] = 0.0
    assert q0.loss_weights == expected_q0
    assert [q0.scheduler.state_for_epoch(epoch)["alpha"] for epoch in (0, 3, 6)] == [0.25, 0.5, 0.75]

    q3 = resolve_curriculum_run(
        _config(
            tmp_path,
            "Q3",
            selected_consumer_json=str(selected),
            student_warm_start_checkpoint="a0.pt",
        )
    )
    assert {q3.scheduler.state_for_epoch(epoch)["alpha"] for epoch in range(7)} == {0.75}
    assert q3.selected_consumer_id == "Ofull"

    with pytest.raises(ValueError, match="loss weights are fixed"):
        resolve_curriculum_run(
            _config(
                tmp_path,
                "Q0",
                selected_consumer_json=str(selected),
                student_warm_start_checkpoint="a0.pt",
                loss_weight_overrides={"field": 0.9},
            )
        )


def test_step7_selected_quarter_endpoint_does_not_ramp_past_selector_choice(tmp_path: Path):
    selected = _selected_consumer(tmp_path / "selected_consumer.json", endpoint=0.25)
    resolved = resolve_curriculum_run(
        _config(
            tmp_path,
            "P7a",
            selected_consumer_json=str(selected),
            student_warm_start_checkpoint="a0.pt",
        )
    )
    assert {resolved.scheduler.state_for_epoch(epoch)["alpha"] for epoch in range(7)} == {0.25}


def test_step7_named_logit_only_fallback_disables_oracle_path_and_marks_non_equivalence(tmp_path: Path):
    selected = _selected_consumer(tmp_path / "selected_consumer.json")
    resolved = resolve_curriculum_run(
        _config(
            tmp_path,
            "P2",
            selected_consumer_json=str(selected),
            oracle_logit_only_fallback=True,
        )
    )
    assert resolved.loss_weights["oracle_path"] == 0.0
    assert resolved.oracle_path_fallback_downgrade is True
    assert resolved.recipe_equivalent is False


def test_step7_loss_composes_all_terms_and_detaches_teacher_targets():
    student_logits = torch.tensor([[1.0, -0.5], [-0.2, 0.4]], requires_grad=True)
    oracle_pred = torch.tensor([[0.7, -0.1], [0.1, 0.3]], requires_grad=True)
    oracle_true = torch.tensor([[1.2, -0.2], [-0.4, 0.8]], requires_grad=True)
    pred_fields = torch.tensor([[[0.5, -0.5]], [[0.2, 0.1]]], requires_grad=True)
    effective = pred_fields * 0.5
    gate_loss = torch.tensor(0.3, requires_grad=True)
    output = SimpleNamespace(
        student_logits=student_logits,
        oracle_pred_logits=oracle_pred,
        oracle_true_logits=oracle_true,
        pred_fields_raw=pred_fields,
        pred_fields_effective=effective,
        field_gate_loss=gate_loss,
    )
    batch = {
        "labels": torch.tensor([0, 1]),
        "target_fields": torch.zeros_like(pred_fields),
        "target_mask": torch.tensor([[True], [True]]),
        "raw_mask": torch.tensor([[True], [True]]),
        "offline_teacher_logits": torch.tensor([[0.3, -0.3], [-0.1, 0.2]]),
    }
    loss, diagnostics = compute_curriculum_batch_loss(
        output,
        batch,
        loss_weights={name: 1.0 for name in ("ce", "student_kd", "oracle_path", "field", "gate", "reg")},
        student_kd_source="offline_teacher",
        kd_temperature=2.0,
        field_huber_beta=0.1,
    )
    assert torch.isfinite(loss)
    assert set(diagnostics) >= {
        "cross_entropy",
        "student_kd_loss",
        "oracle_path_kd_loss",
        "field_huber_loss",
        "gate_loss",
        "regularization_loss",
    }
    loss.backward()
    assert student_logits.grad is not None
    assert oracle_pred.grad is not None
    assert oracle_true.grad is None
    assert pred_fields.grad is not None


def test_step7_validation_coverage_fails_closed():
    ok, reason = curriculum_train_module._coverage_valid(
        {"n_jets": 98, "loss": 1.0},
        expected=100,
        required_fraction=0.99,
    )
    assert ok is False
    assert "98/100" in reason


class FakePart(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = torch.nn.Conv1d(input_dim, num_classes, kernel_size=1)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        values = self.proj(features)
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


class TinyDataset:
    def __init__(self, n: int = 4) -> None:
        self.tokens = np.zeros((n, 3, 14), dtype=np.float32)
        self.tokens[:, :, 0] = np.asarray([5.0, 3.0, 1.0], dtype=np.float32)
        self.tokens[:, :, 3] = self.tokens[:, :, 0] * 1.1
        self.tokens[:, :, 5] = 1.0
        self.mask = np.ones((n, 3), dtype=bool)
        self.labels = np.arange(n, dtype=np.int64) % 3
        self.target_fields = np.zeros((n, 3, 2), dtype=np.float32)
        self.target_fields[:, :, 0] = 0.2
        self.target_mask = self.mask.copy()
        self.field_names = FIELD_NAMES
        self.field_groups = FIELD_GROUPS
        self.metadata = {
            "alignment_report": {
                "source_manifest_hash": "manifest",
                "hlt_content_hash": "hlt",
                "target_content_hash": "target",
                "offline_content_hash": "offline",
                "jet_identity_hash": "ids",
            },
            "target_metadata": {"normalization": "normalized"},
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": self.labels[index],
            "indices": np.int64(index),
            "target_fields": self.target_fields[index],
            "target_mask": self.target_mask[index],
        }


def _tiny_joint() -> LocalResidualFieldCurriculumJointModel:
    reco = LocalResidualFieldReconstructorConfig(
        field_dim=2,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        d_model=8,
        num_heads=2,
        num_layers=1,
        context_layers=1,
        dropout=0.0,
        attention_dropout=0.0,
        max_particles=4,
    )
    student_config = LocalResidualFieldTaggerConfig(
        num_classes=3,
        field_dim=2,
        field_source=RESIDUAL_FIELD_SOURCE_ZERO,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
    )
    student = LocalResidualFieldAugmentedParT(
        student_config,
        part_model=FakePart(len(PF_FEATURE_NAMES) + 2, 3),
    )
    return LocalResidualFieldCurriculumJointModel(
        LocalResidualFieldCurriculumJointConfig(
            reconstructor_config=reco,
            student_config=student_config,
            field_gate_mode=FIELD_GATE_MODE_NONE,
            gate_reliability_loss_weight=0.0,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        student=student,
    )


def test_step7_tiny_p0_training_writes_oracle_free_checkpoint_and_reports(tmp_path: Path):
    dataset = TinyDataset()
    config = _config(
        tmp_path,
        "P0",
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        amp=False,
        early_stop_patience=-1,
    )
    report = train_local_residual_field_curriculum(
        config,
        model=_tiny_joint(),
        datasets={"model_train": dataset, "model_val": dataset, "stack_val": dataset},
    )

    assert report["ok"] is True
    assert report["runtime_inputs"] == "HLT_only"
    assert report["uses_teacher_logits_at_runtime"] is False
    assert report["deployable"] is True
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["contract"] == LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT
    assert checkpoint["oracle_consumer_included"] is False
    assert "oracle_consumer_state_dict" not in checkpoint
    curves = json.loads((Path(config.output_dir) / "training_curves.json").read_text(encoding="utf-8"))
    assert curves["epochs"][0]["schedule"]["loss_weights"] == PILOT_LOSS_WEIGHTS["P0"]
    assert curves["epochs"][0]["model_val"]["valid_fraction"] == 1.0


def test_step7_cli_exposes_all_required_training_inputs():
    script = ROOT / "scripts" / "train_local_residual_field_curriculum_student.py"
    spec = spec_from_file_location("curriculum_step7_cli", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(
        [
            "--run-id", "Q3",
            "--output-dir", "out",
            "--hlt-cache-dir", "hlt",
            "--target-cache-dir", "targets",
            "--selected-consumer-json", "selected_consumer.json",
            "--oracle-teacher-checkpoint", "oracle.pt",
            "--oracle-teacher-logits-dir", "oracle_logits",
            "--offline-teacher-logits-dir", "offline_logits",
            "--student-warm-start-checkpoint", "a0.pt",
            "--predictor-warm-start-checkpoint", "predictor.pt",
            "--evaluate-final-test",
            "--confirm-final-test",
        ]
    )
    assert args.run_id == "Q3"
    assert args.selected_consumer_json == "selected_consumer.json"
    assert args.oracle_teacher_checkpoint == "oracle.pt"
    assert args.confirm_final_test is True
