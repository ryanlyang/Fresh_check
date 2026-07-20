from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.fusion import PredictionBlock, load_prediction_block, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FIELD_GATE_MODE_NONE,
    LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
    LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,
    RESIDUAL_FIELD_SOURCE_ZERO,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldCurriculumJointConfig,
    LocalResidualFieldCurriculumJointModel,
    LocalResidualFieldFusionConfig,
    LocalResidualFieldPredictionConfig,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldTaggerConfig,
    cache_local_residual_field_tagger_predictions,
    run_local_residual_field_fusion,
)
import teacher_logit_reco.local_particle_residual_field.fusion as fusion_module


ROOT = Path(__file__).resolve().parents[1]
FIELD_NAMES = ("field.a", "field.b")
FIELD_GROUPS = {"pair": [0, 1]}


class FakePart(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Conv1d(len(PF_FEATURE_NAMES) + 2, 3, kernel_size=1)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        values = self.proj(features)
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def _joint_model() -> LocalResidualFieldCurriculumJointModel:
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
    student = LocalResidualFieldAugmentedParT(student_config, part_model=FakePart())
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


def _hlt_view(split: str, n: int = 5) -> JetView:
    labels = np.arange(n, dtype=np.int64) % 3
    tokens = np.zeros((n, 4, 14), dtype=np.float32)
    tokens[:, :3, 0] = np.asarray([6.0, 3.0, 1.0], dtype=np.float32)
    tokens[:, :3, 3] = tokens[:, :3, 0] * 1.1
    tokens[:, :3, 5] = 1.0
    mask = np.zeros((n, 4), dtype=bool)
    mask[:, :3] = True
    ids = tuple(JetIdentity(file=f"{split}.root", entry=index, label=int(label)) for index, label in enumerate(labels))
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"hlt-{split}",
            "jet_identity_hash": f"ids-{split}",
            "source_manifest_hash": "manifest",
        },
    )


def test_step8_curriculum_prediction_is_hlt_only_and_records_component_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = _joint_model()
    checkpoint = tmp_path / "P7a" / "best_model_val.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
            "oracle_consumer_included": False,
            "metadata": {
                "run_id": "P7a",
                "teacher_used_during_training": "Ofull",
                "runtime_inputs": "HLT_only",
            },
        },
        checkpoint,
    )
    monkeypatch.setattr(
        LocalResidualFieldCurriculumJointModel,
        "from_deployable_checkpoint",
        classmethod(lambda cls, payload, device="cpu": model.to(device)),
    )
    monkeypatch.setattr(
        fusion_module,
        "load_cached_hlt_view",
        lambda cache_dir, split, verify_hash=True: _hlt_view(str(split)),
    )

    report = cache_local_residual_field_tagger_predictions(
        LocalResidualFieldPredictionConfig(
            checkpoint=str(checkpoint),
            prediction_dir=str(tmp_path / "predictions"),
            model_name="P7a",
            hlt_cache_dir="hlt-only",
            target_cache_dir=None,
            splits=("stack_train", "stack_val", "final_test"),
            batch_size=2,
            amp=False,
            confirm_final_test=True,
        )
    )

    block = load_prediction_block(tmp_path / "predictions", "P7a", "final_test")
    metadata = block.metadata
    assert report["curriculum_deployable_checkpoint"] is True
    assert report["student_checkpoint_hash"] == report["predictor_checkpoint_hash"]
    assert metadata["student_checkpoint_hash"] == report["checkpoint_hash"]
    assert metadata["predictor_checkpoint_hash"] == report["checkpoint_hash"]
    assert metadata["component_checkpoint_storage"] == "joint_deployable_checkpoint"
    assert metadata["teacher_used_during_training"] == "Ofull"
    assert metadata["runtime_inputs"] == "HLT_only"
    assert metadata["uses_true_fields"] is False
    assert metadata["uses_offline_particles"] is False
    assert metadata["uses_teacher_logits_at_runtime"] is False
    assert metadata["deployable"] is True
    assert metadata["dataset_metadata"]["target_fields_present"] is False


def _write_prediction(
    root: Path,
    *,
    model_name: str,
    split: str,
    student_hash: str,
    predictor_hash: str | None,
) -> None:
    labels = np.arange(9, dtype=np.int64) % 3
    logits = np.zeros((9, 3), dtype=np.float32)
    logits[np.arange(9), labels] = 2.0
    save_prediction_block(
        PredictionBlock(
            model_name=model_name,
            split=split,
            logits=logits,
            probs=np.zeros_like(logits),
            labels=labels,
            jet_ids=[JetIdentity(file=f"{split}.root", entry=index, label=int(label)) for index, label in enumerate(labels)],
            metadata={
                "runtime_inputs": "HLT_only",
                "uses_true_fields": False,
                "uses_offline_particles": False,
                "uses_teacher_logits_at_runtime": False,
                "deployable": True,
                "selection_allowed": split != "final_test",
                "student_checkpoint_hash": student_hash,
                "predictor_checkpoint_hash": predictor_hash,
                "teacher_used_during_training": "Ofull" if model_name == "P7a" else None,
            },
        ),
        root,
        overwrite=True,
    )


def test_step8_fusion_preflights_every_selected_member_before_writing_outputs(tmp_path: Path):
    prediction_dir = tmp_path / "predictions"
    for split in ("stack_train", "stack_val", "final_test"):
        _write_prediction(prediction_dir, model_name="A0", split=split, student_hash="a0", predictor_hash=None)
    for split in ("stack_train", "stack_val"):
        _write_prediction(prediction_dir, model_name="P7a", split=split, student_hash="p7a", predictor_hash="p7a")
    output_dir = tmp_path / "fusion"

    with pytest.raises(FileNotFoundError, match="P7a"):
        run_local_residual_field_fusion(
            LocalResidualFieldFusionConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(output_dir),
                groups={"G0": ("A0", "P7a")},
                fusion_modes=(LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,),
                confirm_final_test=True,
            )
        )

    assert not output_dir.exists()


def test_step8_fusion_carries_member_runtime_and_checkpoint_provenance(tmp_path: Path):
    prediction_dir = tmp_path / "predictions"
    for split in ("stack_train", "stack_val", "final_test"):
        _write_prediction(prediction_dir, model_name="A0", split=split, student_hash="a0", predictor_hash=None)
        _write_prediction(prediction_dir, model_name="P7a", split=split, student_hash="p7a", predictor_hash="p7a")

    report = run_local_residual_field_fusion(
        LocalResidualFieldFusionConfig(
            prediction_dir=str(prediction_dir),
            output_dir=str(tmp_path / "fusion"),
            groups={"G0": ("A0", "P7a")},
            fusion_modes=(LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,),
            confirm_final_test=True,
        )
    )

    assert report["selected_member_predictions_complete"] is True
    provenance = report["groups"]["G0"]["member_provenance"]
    assert provenance["runtime_inputs"] == "HLT_only"
    assert provenance["student_checkpoint_hashes"] == {"A0": "a0", "P7a": "p7a"}
    assert provenance["predictor_checkpoint_hashes"] == {"A0": None, "P7a": "p7a"}
    final_metrics = report["groups"]["G0"]["fusion_modes"][LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN]["metrics"]["final_test"]
    assert final_metrics["deployable"] is True
    assert final_metrics["runtime_inputs"] == "HLT_only"
    assert final_metrics["selection_allowed"] is False


def test_step8_prediction_cli_does_not_require_target_cache_for_curriculum():
    script = ROOT / "scripts" / "predict_local_residual_field_tagger.py"
    spec = spec_from_file_location("curriculum_step8_prediction_cli", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(
        [
            "--checkpoint", "P7a/best_model_val.pt",
            "--prediction-dir", "predictions",
            "--model-name", "P7a",
            "--hlt-cache-dir", "hlt",
            "--splits", "stack_val",
        ]
    )
    assert args.target_cache_dir == ""
