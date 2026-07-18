from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    AdaptiveBinaryReconstructorModel,
    build_variant_hierarchy_aware_tagger,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.checkpoints import (
    ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT,
    build_compact_selected_checkpoint,
    load_torch_checkpoint,
    require_exact_resume_checkpoint,
    selected_checkpoint_provenance,
    selected_model_state,
    validate_compact_selected_checkpoint,
    write_selected_checkpoint,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    initialize_storage_accounting,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.training import (
    ABPH_RECONSTRUCTOR_MODULE_GROUPS,
    CyclingSequenceBatchSource,
    OptimizerGroupPolicy,
    ReconstructorCurriculumConfig,
    ReconstructorStepResult,
    ReconstructorTrainerConfig,
    train_reconstructor_curriculum,
)


FORBIDDEN = {
    "online_model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "ema_state_dict",
    "rng_state",
    "train_source_state_dict",
    "distributed_checkpoint_state",
}


def _compact(model: torch.nn.Module, *, role: str = "best_model_val"):
    return build_compact_selected_checkpoint(
        model_state_dict=model.state_dict(),
        checkpoint_role=role,
        model_metadata={"model_class": model.__class__.__qualname__},
        resolved_variant_config={"variant": "test"},
        resolved_variant_config_hash="config-hash",
        validation={"selection_score": 0.25, "n_jets": 8},
        provenance={"manifest_hash": "manifest", "hlt_content_hash": "hlt"},
        runtime_contracts={"runtime": "test"},
        schedule_contracts={"schedule": "test"},
    )


@pytest.mark.parametrize(
    "family,model_factory",
    [
        (
            "B",
            lambda: AdaptiveBinaryReconstructorModel(
                hierarchy_names=("exclusive_kt",),
                variant_name="B1_semantic_query_root",
                smoke=True,
            ),
        ),
        (
            "C",
            lambda: AdaptiveBinaryReconstructorModel(
                hierarchy_names=("exclusive_kt",),
                variant_name="C1_kt_2",
                smoke=True,
            ),
        ),
        (
            "D",
            lambda: AdaptiveBinaryReconstructorModel(
                hierarchy_names=("exclusive_kt",),
                variant_name="D1_kt32_mh4_particles",
                smoke=True,
            ),
        ),
        (
            "E",
            lambda: build_variant_hierarchy_aware_tagger(
                "E5_kt32_mh4_dualcross", smoke=True
            ),
        ),
        (
            "F",
            lambda: build_variant_hierarchy_aware_tagger(
                "F0_ce_reco_primary", smoke=True
            ),
        ),
    ],
)
def test_compact_selected_strictly_reconstructs_campaign_families(
    family, model_factory
):
    source = model_factory()
    if family in {"E", "F"}:
        for module in source.modules():
            if isinstance(module, torch.nn.LazyLinear) and module.has_uninitialized_params():
                module.initialize_parameters(torch.zeros(1, 5))
    payload = _compact(source)
    restored = model_factory()
    if family in {"E", "F"}:
        for module in restored.modules():
            if isinstance(module, torch.nn.LazyLinear) and module.has_uninitialized_params():
                module.initialize_parameters(torch.zeros(1, 5))
    restored.load_state_dict(selected_model_state(payload), strict=True)
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert not FORBIDDEN.intersection(payload)
    assert payload["checkpoint_contract"] == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT


def test_compaction_preserves_warm_start_tensors_and_prediction_logits(tmp_path):
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 7), torch.nn.GELU(), torch.nn.Linear(7, 3)
    ).eval()
    inputs = torch.randn(11, 5)
    legacy = {
        "checkpoint_contract": "legacy_full",
        "checkpoint_role": "best_model_val",
        "model_state_dict": model.state_dict(),
        "online_model_state_dict": {
            name: value + 1.0 for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": {"state": {1: "large"}},
        "scaler_state_dict": {"scale": 1024},
        "ema_state_dict": {"shadow": model.state_dict()},
    }
    compact = _compact(model)
    legacy_target = torch.nn.Sequential(
        torch.nn.Linear(5, 7), torch.nn.GELU(), torch.nn.Linear(7, 3)
    ).eval()
    compact_target = torch.nn.Sequential(
        torch.nn.Linear(5, 7), torch.nn.GELU(), torch.nn.Linear(7, 3)
    ).eval()
    legacy_target.load_state_dict(legacy["model_state_dict"], strict=True)
    compact_target.load_state_dict(compact["model_state_dict"], strict=True)
    assert all(
        torch.equal(legacy["model_state_dict"][name], compact["model_state_dict"][name])
        for name in legacy["model_state_dict"]
    )
    assert torch.equal(legacy_target(inputs), compact_target(inputs))

    path = tmp_path / "best_model_val.pt"
    write_selected_checkpoint(path, compact)
    loaded = load_torch_checkpoint(path)
    assert loaded["content_hash"] == compact["content_hash"]
    assert selected_checkpoint_provenance(path, loaded)["exact_resume"] is False


def test_component_states_are_hashed_once_and_tampering_fails():
    tagger = torch.nn.Linear(3, 2)
    reconstructor = torch.nn.Linear(4, 3)
    payload = build_compact_selected_checkpoint(
        model_state_dict=tagger.state_dict(),
        component_state_dicts={"reconstructor": reconstructor.state_dict()},
        checkpoint_role="best_model_val",
        model_metadata={"kind": "joint"},
        resolved_variant_config={"variant": "F0"},
        resolved_variant_config_hash="f0-config",
        validation={"loss": 1.0},
        provenance={"manifest_hash": "m"},
    )
    assert set(payload["component_state_dicts"]) == {"reconstructor"}
    assert "reconstructor_state_dict" not in payload
    validate_compact_selected_checkpoint(payload)
    payload["component_state_dicts"]["reconstructor"]["weight"][0, 0] += 1.0
    with pytest.raises(ValueError, match="component-state hash mismatch"):
        validate_compact_selected_checkpoint(payload)


def test_compact_selected_checkpoint_can_never_claim_exact_resume():
    payload = _compact(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="warm starts, not exact optimizer resumes"):
        require_exact_resume_checkpoint(payload)
    assert payload["exact_resume_supported"] is False
    assert payload["restart_semantics"] == "selected_weights_warm_start_not_exact_resume"


def test_streaming_write_is_quota_accounted(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    initialize_storage_accounting(root, profile="streaming_30gb_v1")
    monkeypatch.setenv("ABPH_STORAGE_PROFILE", "streaming_30gb_v1")
    model = torch.nn.Linear(2, 2)
    payload = _compact(model)
    path = root / "runs" / "B1" / "best_model_val.pt"
    receipt = write_selected_checkpoint(
        path,
        payload,
        campaign_root=root,
        artifact_role="selected_reconstructor_checkpoint",
        run_id="B1",
    )
    assert receipt["status"] == "committed"
    assert receipt["actual_bytes"] == path.stat().st_size
    assert load_torch_checkpoint(path)["content_hash"] == payload["content_hash"]


class _StreamingTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS:
            setattr(self, name, torch.nn.Linear(1, 1, bias=False))

    def module_groups(self):
        return {name: getattr(self, name) for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS}


def _streaming_step(model, batch, context):
    evidence = model.hlt_encoder(batch["x"])
    values = {
        "root": model.root(evidence).square().mean(),
        **{
            f"group_{capacity}": getattr(model, f"hierarchy_{capacity}")(
                evidence
            ).square().mean()
            for capacity in (2, 4, 8, 16, 32)
        },
        "particle": model.renderer(evidence).square().mean(),
        "distribution": model.distribution(evidence).square().mean(),
    }
    active = values[f"group_{max(2, context.curriculum.active_capacity)}"]
    return ReconstructorStepResult(
        loss_terms={
            **values,
            "topology": active,
            "frontier": active,
            "particle_feature": values["particle"],
            "calibration": values["distribution"],
            "auxiliary": values["particle"],
        },
        metrics={},
        batch_size=int(batch["x"].shape[0]),
    )


def test_streaming_curriculum_keeps_resume_and_stage_handoffs_in_ram(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    output = root / "runs" / "B1_semantic_query_root"
    ram = tmp_path / "ram"
    (ram / "checkpoints").mkdir(parents=True)
    initialize_storage_accounting(root, profile="streaming_30gb_v1")
    monkeypatch.setenv("ABPH_STORAGE_PROFILE", "streaming_30gb_v1")
    monkeypatch.setenv("ABPH_RAM_WORKSPACE", str(ram))
    model = _StreamingTinyModel()
    curriculum = ReconstructorCurriculumConfig(
        root_updates=1,
        hierarchy_updates_per_depth=1,
        renderer_updates=1,
        distribution_updates=1,
        evaluation_interval=1,
        root_patience_evaluations=20,
        hierarchy_patience_evaluations=20,
        renderer_patience_evaluations=20,
        distribution_patience_evaluations=20,
    )
    config = ReconstructorTrainerConfig(
        output_dir=str(output),
        seed=7,
        device="cpu",
        amp=False,
        gradient_accumulation_steps=1,
        root_hierarchy_effective_batch_size=2,
        renderer_distribution_effective_batch_size=2,
        ema_decay=0.0,
        curriculum=curriculum,
    )
    source = CyclingSequenceBatchSource([{"x": torch.ones(2, 1)}])
    policies = {
        name: OptimizerGroupPolicy(peak_lr=0.01, weight_decay=0.0)
        for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS
    }
    report = train_reconstructor_curriculum(
        model,
        model.module_groups(),
        source,
        lambda: [{"x": torch.ones(2, 1)}],
        _streaming_step,
        config,
        provenance={"manifest_hash": "m", "hlt_content_hash": "h"},
        optimizer_policies=policies,
    )
    assert report["ok"] is True
    assert not (output / "last.pt").exists()
    assert not list((ram / "checkpoints").rglob("*.pt"))
    selected = load_torch_checkpoint(output / "best_model_val.pt")
    assert selected["checkpoint_contract"] == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT
    assert not FORBIDDEN.intersection(selected)
    assert report["checkpoint_storage"]["failed_allocation_loses_optimizer_state"]
    assert all(
        row.get("ephemeral_checkpoint_deleted") is True
        for row in report["best_by_stage"].values()
    )
