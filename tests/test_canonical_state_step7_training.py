import pytest
import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.local_compression_part.config import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_LOSS_CONTRACT,
    CANONICAL_STATE_TRAINING_CONTRACT,
    SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
    SCHEDULE_FROM_SCRATCH_PART_BASELINE,
    SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
    SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
    STATE_CONTEXT_PHI_HLT,
    CanonicalStateConditionedParT,
    CanonicalStateLossWeights,
    CanonicalStateTaggerConfig,
    apply_canonical_state_train_phase,
    assert_teacher_free_final_test,
    build_canonical_state_training_schedule,
    canonical_state_nonfinite_batch_report,
    canonical_state_optimizer_group_specs,
    canonical_state_schedule_requires_warm_start,
    canonical_state_should_skip_batch,
    compute_canonical_state_losses,
    default_canonical_jet_state_layout,
    normalize_canonical_state_training_schedule,
    teacher_logits_allowed_for_split,
)
from teacher_logit_reco.canonical_state.run_variants import _state_prediction_metrics


class _FakeParT(torch.nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES),
        embed_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.mod = torch.nn.Module()
        self.mod.embed = torch.nn.Linear(feature_dim, embed_dim)
        self.mod.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, points, features, lorentz_vectors, mask):  # noqa: ANN001
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        embeddings = self.mod.embed(rows)
        valid = mask[:, 0, :].bool()[:, :, None].to(dtype=embeddings.dtype)
        pooled = (embeddings * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.mod.fc(pooled)


def _tagger() -> CanonicalStateConditionedParT:
    return CanonicalStateConditionedParT(
        CanonicalStateTaggerConfig(
            mode=STATE_CONTEXT_PHI_HLT,
            part_embed_dim=32,
            state_dim=32,
            state_layers=1,
            state_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
            max_constits=16,
            predictor_config={
                "d_model": 32,
                "num_heads": 4,
                "particle_encoder_layers": 1,
                "decoder_layers": 1,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "max_particles": 16,
            },
        ),
        part_model=_FakeParT(),
    )


def _loss_inputs():
    layout = default_canonical_jet_state_layout()
    torch.manual_seed(123)
    logits = torch.randn(4, 10)
    labels = torch.tensor([0, 2, 4, 8])
    teacher_logits = logits + 0.2 * torch.randn_like(logits)
    phi_hlt = torch.randn(4, layout.k_state, layout.d_phi) * 0.04
    true_delta = torch.randn_like(phi_hlt) * 0.01
    phi_off = phi_hlt + true_delta
    delta_pred = true_delta * 0.75
    log_sigma = torch.zeros_like(delta_pred)
    state_mask = torch.ones(4, layout.k_state, dtype=torch.bool)
    state_mask[0, -3:] = False
    return logits, labels, teacher_logits, phi_hlt, phi_off, delta_pred, log_sigma, state_mask


def test_loss_terms_active_and_inactive_by_weights() -> None:
    logits, labels, teacher_logits, phi_hlt, phi_off, delta_pred, log_sigma, state_mask = _loss_inputs()

    ce_only = compute_canonical_state_losses(
        logits=logits,
        labels=labels,
        weights=CanonicalStateLossWeights(ce=1.0),
    )

    assert ce_only.diagnostics["contract"] == CANONICAL_STATE_LOSS_CONTRACT
    assert set(ce_only.terms) == {"ce"}
    assert ce_only.diagnostics["active_terms"] == ["ce"]

    full = compute_canonical_state_losses(
        logits=logits,
        labels=labels,
        teacher_logits=teacher_logits,
        phi_hlt=phi_hlt,
        phi_off=phi_off,
        delta_phi_pred=delta_pred,
        log_sigma=log_sigma,
        state_mask=state_mask,
        weights=CanonicalStateLossWeights(
            ce=1.0,
            state_huber=1.0,
            state_l1=0.25,
            logit_kd=0.5,
            delta_norm=0.01,
            smoothness=0.02,
            uncertainty_state=0.1,
        ),
    )

    assert {
        "ce",
        "state_huber",
        "state_l1",
        "logit_kd",
        "delta_norm",
        "smoothness",
        "uncertainty_state",
    }.issubset(set(full.terms))
    assert torch.isfinite(full.total)
    assert full.diagnostics["teacher_logits_used"] is True


def test_state_prediction_metrics_ignore_invalid_state_slots() -> None:
    layout = default_canonical_jet_state_layout()
    phi_hlt = torch.zeros(1, layout.k_state, layout.d_phi)
    phi_off = torch.zeros_like(phi_hlt)
    delta = torch.zeros_like(phi_hlt)
    delta[:, -1, :] = 1000.0
    state_mask = torch.ones(1, layout.k_state, dtype=torch.bool)
    state_mask[:, -1] = False

    metrics = _state_prediction_metrics(delta, phi_hlt, phi_off, state_mask)

    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["valid_state_tokens_mean"] == pytest.approx(float(layout.k_state - 1))


def test_teacher_free_final_test_loss_and_assertion() -> None:
    logits, labels, teacher_logits, *_ = _loss_inputs()
    assert teacher_logits_allowed_for_split("model_val") is True
    assert teacher_logits_allowed_for_split("final_test") is False
    assert teacher_logits_allowed_for_split("final_test", allow_final_test_teacher=True) is True

    with pytest.raises(ValueError, match="teacher logits"):
        compute_canonical_state_losses(
            logits=logits,
            labels=labels,
            teacher_logits=teacher_logits,
            split="final_test",
            weights=CanonicalStateLossWeights(ce=1.0, logit_kd=1.0),
        )
    with pytest.raises(ValueError, match="teacher-free"):
        assert_teacher_free_final_test(split="final_test", teacher_logits=teacher_logits)
    assert_teacher_free_final_test(split="final_test", teacher_logits=teacher_logits, allow_teacher_diagnostics=True)


def test_loss_requires_targets_when_terms_are_enabled() -> None:
    logits, labels, *_ = _loss_inputs()
    with pytest.raises(ValueError, match="delta_phi_pred"):
        compute_canonical_state_losses(
            logits=logits,
            labels=labels,
            weights=CanonicalStateLossWeights(ce=0.0, state_huber=1.0),
        )
    with pytest.raises(ValueError, match="teacher_logits"):
        compute_canonical_state_losses(
            logits=logits,
            labels=labels,
            weights=CanonicalStateLossWeights(ce=0.0, logit_kd=1.0),
        )


def test_training_schedules_and_warm_start_policy() -> None:
    schedule = build_canonical_state_training_schedule(
        SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
        total_epochs=8,
        warmup_epochs=2,
        adapter_warmup_epochs=2,
        loss_weights=CanonicalStateLossWeights(ce=1.0, state_huber=0.5),
    )
    assert schedule.to_dict()["contract"] == CANONICAL_STATE_TRAINING_CONTRACT
    assert schedule.phase_for_epoch(1).name == "state_predictor_pretrain"
    assert schedule.phase_for_epoch(3).name == "tagger_adapter_warmup"
    assert schedule.phase_for_epoch(5).name == "joint_finetune_after_pretrain"
    assert canonical_state_schedule_requires_warm_start(SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE) is True
    assert canonical_state_schedule_requires_warm_start(SCHEDULE_FROM_SCRATCH_CANONICAL_STATE) is False
    assert canonical_state_schedule_requires_warm_start(SCHEDULE_FROM_SCRATCH_PART_BASELINE) is False
    assert normalize_canonical_state_training_schedule("pretrain") == SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER


def test_freeze_schedule_changes_trainable_params_and_optimizer_groups() -> None:
    model = _tagger()
    schedule = build_canonical_state_training_schedule(SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE, warmup_epochs=1)

    warmup_report = apply_canonical_state_train_phase(model, schedule.phase_for_epoch(1))
    warmup_groups = warmup_report["module_group_trainability"]
    assert warmup_groups["part_model"]["trainable"] == 0
    assert warmup_groups["part_head"]["trainable"] > 0
    assert warmup_groups["state_adapter"]["trainable"] > 0
    warmup_group_names = {group["name"] for group in canonical_state_optimizer_group_specs(model, schedule.phase_for_epoch(1))}
    assert "part_model" not in warmup_group_names
    assert "part_head" in warmup_group_names

    finetune_report = apply_canonical_state_train_phase(model, schedule.phase_for_epoch(2))
    finetune_groups = finetune_report["module_group_trainability"]
    assert finetune_groups["part_model"]["trainable"] > 0
    assert finetune_groups["state_predictor"]["trainable"] > 0
    finetune_group_names = {group["name"] for group in canonical_state_optimizer_group_specs(model, schedule.phase_for_epoch(2))}
    assert "part_model" in finetune_group_names


def test_from_scratch_schedule_reports_no_warm_start() -> None:
    schedule = build_canonical_state_training_schedule(SCHEDULE_FROM_SCRATCH_CANONICAL_STATE)
    phase = schedule.phase_for_epoch(1)
    assert schedule.requires_warm_start is False
    assert phase.requires_warm_start is False


def test_nonfinite_batch_guard() -> None:
    good = torch.ones(2, 3)
    bad = torch.tensor([1.0, float("nan")])
    good_report = canonical_state_nonfinite_batch_report(logits=good, loss=torch.tensor(1.0))
    bad_report = canonical_state_nonfinite_batch_report(logits=good, loss=bad)
    assert good_report["ok"] is True
    assert bad_report["skip_batch"] is True
    assert bad_report["nonfinite_fields"] == ["loss"]
    assert canonical_state_should_skip_batch(loss=bad, logits=good) is True
    assert canonical_state_should_skip_batch(loss=torch.tensor(1.0), logits=good) is False
